#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1. Load metabolomics data, correlation matrix, and a pre-trained model.
2. Fine-tune independently for each disease (binary classification: 1=disease, 0=healthy).
3. Record detailed logs and training progress.
4. Compute accuracy, precision, F1, and AUC metrics, and generate a TSNE plot.
5. Save each participant's prediction results and average attention weights, and generate a bar plot for attention features.
"""

import os
import glob
import math
import random
import warnings
import logging
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import BertConfig, get_cosine_schedule_with_warmup

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_count = torch.cuda.device_count()
    logging.info(f"Using device: {device}, GPU count: {gpu_count}")
else:
    device = torch.device("cpu")
    gpu_count = 0
    logging.info("Training on CPU")

# Create the main output directory for all results (with timestamp)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
main_output_dir = os.path.join("your/path", f"finetune_{timestamp}")
os.makedirs(main_output_dir, exist_ok=True)
logging.info(f"All results will be saved in directory: {main_output_dir}")

#############################################
# 1. Data Loading and Preprocessing Section
#############################################

expression_data_path = os.path.join("your/path", "metabolomic.csv")
correlation_matrix_path = os.path.join("your/path", "metabolomic_correlation_matrix.csv")
healthy_labels_path = os.path.join("your/path", "healthy_eids.csv")
pretrained_model_path = os.path.join("your/path", "best_metabolite_bert_model.pt")
labels_dir = os.path.join("your/path", "labels")

logging.info("Loading metabolite expression data...")
expression_data = pd.read_csv(expression_data_path)
if 'eid' not in expression_data.columns:
    logging.error("The expression data is missing the 'eid' column!")
    exit(1)

logging.info(f"Expression data shape: {expression_data.shape}")

logging.info("Loading metabolite correlation matrix...")
correlation_matrix = pd.read_csv(correlation_matrix_path, index_col=0)
logging.info(f"Original correlation matrix shape: {correlation_matrix.shape}")

all_metabolite_names = list(expression_data.columns)
all_metabolite_names.remove('eid')

common_metabolites = sorted(
    list(set(all_metabolite_names) & set(correlation_matrix.index) & set(correlation_matrix.columns))
)
logging.info(f"Number of common metabolites (intersection): {len(common_metabolites)}")

expression_data = expression_data[['eid'] + common_metabolites]
logging.info(f"Filtered expression data shape: {expression_data.shape}")

correlation_matrix = correlation_matrix.loc[common_metabolites, common_metabolites]
logging.info(f"Filtered correlation matrix shape: {correlation_matrix.shape}")

bias_matrix = torch.tensor(correlation_matrix.values, dtype=torch.float32, device=device)
logging.info(f"Tensor shape of the original correlation matrix: {bias_matrix.shape}")
bias_matrix = F.pad(bias_matrix, (1, 0, 1, 0), "constant", 0)
logging.info(f"Tensor shape of correlation matrix after including CLS token: {bias_matrix.shape}")

#############################################
# 2. Define the PyTorch Dataset
#############################################
class FineTuneDataset(Dataset):
    def __init__(self, X, y, eids):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.eids = eids

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
            self.eids[idx]
        )

#############################################
# 3. Define the Model Architecture
#############################################

class CustomBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            bias_matrix_chunk=None,
            bias_coef=None
    ):
        mixed_query_layer = self.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)
        else:
            past_key_value = None

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if bias_matrix_chunk is not None and bias_coef is not None:
            bias = bias_matrix_chunk.unsqueeze(0).unsqueeze(0) * bias_coef
            bias = bias.expand(attention_scores.size())
            attention_scores = attention_scores + bias

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)

        return outputs


class CustomBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = CustomBertSelfAttention(config)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            bias_matrix_chunk=None,
            bias_coef=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            bias_matrix_chunk=bias_matrix_chunk,
            bias_coef=bias_coef
        )

        attention_output = self.output(self_outputs[0])
        attention_output = self.dropout(attention_output)
        attention_output = self.LayerNorm(attention_output + hidden_states)

        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class CustomBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = CustomBertAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = CustomBertAttention(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            bias_matrix_chunk=None,
            bias_coef=None,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            bias_matrix_chunk=bias_matrix_chunk,
            bias_coef=bias_coef
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions=output_attentions,
                bias_matrix_chunk=bias_matrix_chunk,
                bias_coef=bias_coef
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]

        intermediate_output = self.intermediate(attention_output)
        intermediate_output = F.relu(intermediate_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.LayerNorm(layer_output + attention_output)
        outputs = (layer_output,) + outputs

        return outputs


class CustomBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([CustomBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            bias_matrix_chunk=None,
            bias_coef=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i] if head_mask is not None else None,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_values[i] if past_key_values is not None else None,
                output_attentions=output_attentions,
                bias_matrix_chunk=bias_matrix_chunk,
                bias_coef=bias_coef
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_decoder_cache, all_hidden_states, all_attentions] if v is not None)
        return {
            'last_hidden_state': hidden_states,
            'past_key_values': next_decoder_cache,
            'hidden_states': all_hidden_states,
            'attentions': all_attentions,
        }


class CustomBertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = CustomBertEncoder(config)
        self.pooler = nn.AdaptiveAvgPool1d(1)
        self.config = config
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.uniform_(module.weight, -0.1, 0.1)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def get_extended_attention_mask(self, attention_mask, input_shape, device):
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def invert_attention_mask(self, encoder_attention_mask):
        return (1.0 - encoder_attention_mask) * -10000.0

    def get_head_mask(self, head_mask, num_hidden_layers):
        if head_mask is not None:
            head_mask = torch.tensor(head_mask, dtype=torch.float32)
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).expand(num_hidden_layers, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1)
            return head_mask
        return [None] * num_hidden_layers

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=True,
            bias_matrix_chunk=None,
            bias_coef=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        batch_size, seq_length = input_shape[0], input_shape[1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long,
                                         device=inputs_embeds.device if inputs_embeds is not None else device)

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length),
                                        device=inputs_embeds.device if inputs_embeds is not None else device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)

        if encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones((encoder_batch_size, encoder_sequence_length),
                                                    device=inputs_embeds.device if inputs_embeds is not None else device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = inputs_embeds
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            bias_matrix_chunk=bias_matrix_chunk,
            bias_coef=bias_coef,
        )

        sequence_output = encoder_outputs['last_hidden_state']
        pooled_output = self.pooler(sequence_output.transpose(1, 2)).squeeze(-1) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + tuple(
                v for v in [encoder_outputs['past_key_values'],
                            encoder_outputs['hidden_states'],
                            encoder_outputs['attentions']] if v is not None)

        return {
            'last_hidden_state': sequence_output,
            'pooler_output': pooled_output,
            'past_key_values': encoder_outputs['past_key_values'],
            'hidden_states': encoder_outputs['hidden_states'],
            'attentions': encoder_outputs['attentions'],
        }

# ----- Define the BERT Model -----
class MetaboliteBERTModel(nn.Module):
    def __init__(self, hidden_size, num_layers, num_metabolites, bias_matrix):
        super(MetaboliteBERTModel, self).__init__()
        config = BertConfig(
            vocab_size=num_metabolites,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=8,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=num_metabolites + 1,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            is_decoder=False,
        )
        self.bert = CustomBertModel(config)
        # Use Hadamard (element-wise) mapping: assign each metabolite a learnable weight and bias
        self.expr_weight = nn.Parameter(torch.ones(1, num_metabolites, hidden_size))
        self.expr_bias = nn.Parameter(torch.zeros(1, num_metabolites, hidden_size))
        # The [CLS] token is used for global feature summarization
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.output_layer = nn.Linear(hidden_size, 1)
        self.register_buffer('bias_matrix_full', bias_matrix)
        self.bias_coef = nn.Parameter(torch.tensor(0.01))

    def forward(self, expressions, attention_mask):
        # expressions: [batch_size, num_metabolites]
        batch_size, num_metabolites = expressions.size()
        expr_embeds = expressions.unsqueeze(-1) * self.expr_weight + self.expr_bias  # [B, num_metabolites, hidden_size]
        cls_tokens = self.cls_token.expand(batch_size, 1, -1)  # [B, 1, hidden_size]
        embeddings = torch.cat((cls_tokens, expr_embeds), dim=1)  # [B, num_metabolites+1, hidden_size]
        new_attention_mask = torch.cat((torch.ones(batch_size, 1, device=attention_mask.device), attention_mask), dim=1)
        outputs = self.bert(
            inputs_embeds=embeddings,
            attention_mask=new_attention_mask,
            output_attentions=True,
            bias_matrix_chunk=self.bias_matrix_full,
            bias_coef=self.bias_coef
        )

        prediction_scores = self.output_layer(outputs['last_hidden_state'][:, 1:, :]).squeeze(-1)
        attentions = outputs['attentions']
        return prediction_scores, attentions

# ----- Define the Model for Binary Classification Fine-Tuning -----
class MetaboliteBERTForClassification(nn.Module):
    def __init__(self, hidden_size, num_layers, num_metabolites, bias_matrix, pretrained_path=None):
        super(MetaboliteBERTForClassification, self).__init__()
        self.metabolite_model = MetaboliteBERTModel(hidden_size, num_layers, num_metabolites, bias_matrix)
        self.classifier = nn.Linear(hidden_size, 1)
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)

    def load_pretrained(self, pretrained_path):
        logging.info(f"Loading pre-trained model weights from: {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location=device)
        self.metabolite_model.load_state_dict(state_dict, strict=False)

    def forward(self, expressions, attention_mask):
        batch_size = expressions.size(0)
        expr_embeds = expressions.unsqueeze(-1) * self.metabolite_model.expr_weight + self.metabolite_model.expr_bias
        cls_tokens = self.metabolite_model.cls_token.expand(batch_size, 1, -1)
        embeddings = torch.cat((cls_tokens, expr_embeds), dim=1)
        new_attention_mask = torch.cat((torch.ones(batch_size, 1, device=attention_mask.device), attention_mask), dim=1)
        outputs = self.metabolite_model.bert(
            inputs_embeds=embeddings,
            attention_mask=new_attention_mask,
            output_attentions=True,
            bias_matrix_chunk=self.metabolite_model.bias_matrix_full,
            bias_coef=self.metabolite_model.bias_coef
        )
        pooled_output = outputs['pooler_output']
        logits = self.classifier(pooled_output)
        return logits.squeeze(-1), outputs['attentions']

#################################
# 4. Fine-Tuning Training Function
#################################
def train_finetune_for_disease(disease_label_file):
    # Extract the disease name
    disease_basename = os.path.basename(disease_label_file)
    disease_name = disease_basename.split('_labels.csv')[0]
    logging.info("=" * 80)
    logging.info(f"Starting fine-tuning task for: {disease_name}")

    # Create a subdirectory for saving results for this disease
    disease_output_dir = os.path.join(main_output_dir, disease_name)
    os.makedirs(disease_output_dir, exist_ok=True)

    df_disease = pd.read_csv(disease_label_file)
    if 'eid' not in df_disease.columns:
        logging.error(f"'eid' column is missing in {disease_label_file}! Skipping this disease.")
        return None
    df_disease = df_disease[['eid']].drop_duplicates()
    df_disease['label'] = 1

    df_healthy = pd.read_csv(healthy_labels_path)
    if 'eid' not in df_healthy.columns:
        logging.error(f"'eid' column is missing in {healthy_labels_path}!")
        return None
    df_healthy = df_healthy[['eid']].drop_duplicates()
    df_healthy['label'] = 0

    df_all_labels = pd.concat([df_disease, df_healthy], ignore_index=True).drop_duplicates(subset=['eid'])
    logging.info(f"Total number of samples after merging labels: {df_all_labels.shape[0]}")

    df_expr = expression_data.copy()
    df_merged = pd.merge(df_all_labels, df_expr, on='eid', how='inner')
    logging.info(f"Merged sample count: {df_merged.shape} (Original label count: {df_all_labels.shape[0]})")
    if df_merged.shape[0] < 50:
        logging.warning(f"Not enough samples, skipping {disease_name}")
        return None

    X = df_merged[common_metabolites].values
    y = df_merged['label'].values
    eids = df_merged['eid'].tolist()

    X_train, X_val, y_train, y_val, eids_train, eids_val = train_test_split(
        X, y, eids, test_size=0.2, random_state=42, stratify=y
    )
    logging.info(f"Training set samples: {len(y_train)}, Validation set samples: {len(y_val)}")

    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)
    train_std[train_std == 0] = 1.0
    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std
    logging.info("Completed X data normalization.")

    batch_size = 256
    train_dataset = FineTuneDataset(X_train, y_train, eids_train)
    val_dataset = FineTuneDataset(X_val, y_val, eids_val)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # ----- Hyperparameters -----
    hidden_size = 768
    num_layers = 12
    learning_rate_finetune = 5e-5
    num_epochs = 20

    model = MetaboliteBERTForClassification(
        hidden_size,
        num_layers,
        num_metabolites=len(common_metabolites),
        bias_matrix=bias_matrix,
        pretrained_path=pretrained_model_path
    )
    if gpu_count > 1:
        model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate_finetune)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps),
                                                num_training_steps=total_steps)

    # Record data for each epoch
    train_losses = []
    val_losses = []
    best_val_auc = 0.0
    best_epoch = -1
    best_model_state = None

    overall_metrics = {}

    logging.info(f"Starting fine-tuning for {disease_name}...")
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        all_train_labels = []
        all_train_preds = []
        train_progress = tqdm(train_dataloader, desc=f"[{disease_name}] Training Epoch {epoch + 1}/{num_epochs}")
        for batch in train_progress:
            expressions, labels, _ = batch
            expressions = expressions.to(device)
            labels = labels.to(device)

            attention_mask = torch.ones(expressions.size(), dtype=torch.long, device=device)

            optimizer.zero_grad()
            logits, _ = model(expressions, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_train_loss += loss.item() * expressions.size(0)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_train_preds.extend(probs.tolist())
            all_train_labels.extend(labels.detach().cpu().numpy().tolist())
            train_progress.set_postfix(loss=loss.item())
        epoch_train_loss /= len(train_dataset)
        train_losses.append(epoch_train_loss)
        train_pred_label = [1 if p >= 0.5 else 0 for p in all_train_preds]
        train_acc = accuracy_score(all_train_labels, train_pred_label)
        train_precision = precision_score(all_train_labels, train_pred_label, zero_division=0)
        train_recall = recall_score(all_train_labels, train_pred_label, zero_division=0)
        train_f1 = f1_score(all_train_labels, train_pred_label, zero_division=0)

        # ----- Validation Phase -----
        model.eval()
        epoch_val_loss = 0.0
        all_val_labels = []
        all_val_preds = []
        all_val_probs = []
        sum_attention = None
        count_attention = 0

        with torch.no_grad():
            val_progress = tqdm(val_dataloader, desc=f"[{disease_name}] Validation Epoch {epoch + 1}/{num_epochs}")
            for batch in val_progress:
                expressions, labels, _ = batch
                expressions = expressions.to(device)
                labels = labels.to(device)
                attention_mask = torch.ones(expressions.size(), dtype=torch.long, device=device)
                logits, attentions = model(expressions, attention_mask)
                loss = criterion(logits, labels)
                epoch_val_loss += loss.item() * expressions.size(0)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                all_val_probs.extend(probs.tolist())
                all_val_preds.extend([1 if p >= 0.5 else 0 for p in probs.tolist()])
                all_val_labels.extend(labels.detach().cpu().numpy().tolist())

                if attentions:
                    att = attentions[-1]
                    att = att.mean(dim=1)
                    att = att[:, 1:, 1:]
                    if sum_attention is None:
                        sum_attention = att.sum(dim=0).cpu()
                    else:
                        sum_attention += att.sum(dim=0).cpu()
                    count_attention += expressions.size(0)
            epoch_val_loss /= len(val_dataset)
            val_losses.append(epoch_val_loss)

        try:
            val_acc = accuracy_score(all_val_labels, all_val_preds)
            val_precision = precision_score(all_val_labels, all_val_preds, zero_division=0)
            val_recall = recall_score(all_val_labels, all_val_preds, zero_division=0)
            val_f1 = f1_score(all_val_labels, all_val_preds, zero_division=0)
            val_auc = roc_auc_score(all_val_labels, all_val_probs)
        except Exception as e:
            logging.warning(f"Error computing validation metrics: {e}")
            val_acc, val_precision, val_recall, val_f1, val_auc = 0, 0, 0, 0, 0

        logging.info(
            f"[{disease_name}] Epoch {epoch + 1}: Train Loss: {epoch_train_loss:.4f}, "
            f"Train Acc: {train_acc:.4f}, F1: {train_f1:.4f} | Val Loss: {epoch_val_loss:.4f}, "
            f"Val Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            best_model_state = model.state_dict()

    # Save the best model
    if best_model_state is not None:
        best_model_path = os.path.join(disease_output_dir, f"best_finetune_model_{disease_name}.pt")
        torch.save(best_model_state, best_model_path)
        logging.info(f"[{disease_name}] Best model saved at: {best_model_path}")

    results_df = pd.DataFrame({
        "eid": eids_val,
        "True_Label": all_val_labels,
        "Predicted_Probability": all_val_probs,
        "Predicted_Label": all_val_preds
    })
    results_csv_path = os.path.join(disease_output_dir, f"finetune_predictions_{disease_name}.csv")
    results_df.to_csv(results_csv_path, index=False)
    logging.info(f"[{disease_name}] Validation set prediction results saved at {results_csv_path}")

    def get_pooled_output(model, expressions, attention_mask):
        model_to_use = model.module if hasattr(model, "module") else model
        batch_size = expressions.size(0)
        expr_embeds = expressions.unsqueeze(-1) * model_to_use.metabolite_model.expr_weight + model_to_use.metabolite_model.expr_bias
        cls_tokens = model_to_use.metabolite_model.cls_token.expand(batch_size, 1, -1)
        embeddings = torch.cat((cls_tokens, expr_embeds), dim=1)
        new_attention_mask = torch.cat((torch.ones(batch_size, 1, device=attention_mask.device), attention_mask), dim=1)
        outputs = model_to_use.metabolite_model.bert(
            inputs_embeds=embeddings,
            attention_mask=new_attention_mask,
            output_attentions=False,
            bias_matrix_chunk=model_to_use.metabolite_model.bias_matrix_full,
            bias_coef=model_to_use.metabolite_model.bias_coef
        )
        pooled = outputs['pooler_output']
        return pooled

    pooled_features = []
    tsne_labels = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f"[{disease_name}] Extracting TSNE features"):
            expressions, labels, _ = batch
            expressions = expressions.to(device)
            attention_mask = torch.ones(expressions.size(), dtype=torch.long, device=device)
            pooled = get_pooled_output(model, expressions, attention_mask)
            pooled_features.append(pooled.cpu().numpy())
            tsne_labels.extend(labels.cpu().numpy().tolist())
    pooled_features = np.concatenate(pooled_features, axis=0)
    logging.info(f"Pooled feature shape: {pooled_features.shape}")

    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(pooled_features)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=tsne_labels, palette="coolwarm", s=60)
    plt.title(f"TSNE for {disease_name}")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.legend(title="Label")
    tsne_plot_path = os.path.join(disease_output_dir, f"tsne_{disease_name}.pdf")
    plt.tight_layout()
    plt.savefig(tsne_plot_path)
    plt.close()
    logging.info(f"[{disease_name}] TSNE cluster plot saved at {tsne_plot_path}")

    # ----- Interpretability Analysis: Extract Average Attention Weights -----
    if count_attention > 0:
        mean_attention = sum_attention / count_attention  # [num_metabolites, num_metabolites]
        metabolite_importance = mean_attention.sum(dim=0).numpy()
        importance_df = pd.DataFrame({
            "Metabolite": common_metabolites,
            "Attention_Weight": metabolite_importance
        }).sort_values(by="Attention_Weight", ascending=False)
        importance_csv_path = os.path.join(disease_output_dir, f"metabolite_importance_{disease_name}.csv")
        importance_df.to_csv(importance_csv_path, index=False)
        logging.info(f"[{disease_name}] Metabolite attention weights saved at {importance_csv_path}")
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Attention_Weight", y="Metabolite", data=importance_df.head(10), palette="viridis")
        plt.title(f"Top 10 Important Metabolites for {disease_name}")
        plt.xlabel("Attention Weight")
        plt.ylabel("Metabolite")
        plt.tight_layout()
        top10_plot_path = os.path.join(disease_output_dir, f"top10_metabolites_{disease_name}.pdf")
        plt.savefig(top10_plot_path)
        plt.close()
        logging.info(f"[{disease_name}] Top 10 attention feature plot saved at {top10_plot_path}")
    else:
        logging.warning(f"[{disease_name}] No attention weight information collected.")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.eval()
        train_all_eids, train_all_labels, train_all_probs, train_all_preds = [], [], [], []
        train_eval_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
        with torch.no_grad():
            for batch in tqdm(train_eval_dataloader, desc=f"[{disease_name}] Extracting training set predictions"):
                expressions, labels, eids_batch = batch
                expressions = expressions.to(device)
                labels = labels.to(device)
                attention_mask = torch.ones(expressions.size(), dtype=torch.long, device=device)
                logits, _ = model(expressions, attention_mask)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                preds = [1 if p >= 0.5 else 0 for p in probs.tolist()]
                train_all_eids.extend(eids_batch)
                train_all_labels.extend(labels.cpu().numpy().tolist())
                train_all_probs.extend(probs.tolist())
                train_all_preds.extend(preds)
        train_results_df = pd.DataFrame({
            "eid": train_all_eids,
            "True_Label": train_all_labels,
            "Predicted_Probability": train_all_probs,
            "Predicted_Label": train_all_preds
        })
        train_results_csv_path = os.path.join(disease_output_dir, f"finetune_predictions_train_{disease_name}.csv")
        train_results_df.to_csv(train_results_csv_path, index=False)
        logging.info(f"[{disease_name}] Training set prediction results saved at {train_results_csv_path}")

        results_df = results_df.assign(Set='Validation')
        train_results_df = train_results_df.assign(Set='Train')
        combined_df = pd.concat([train_results_df, results_df]).reset_index(drop=True)

        combined_df['eid'] = combined_df['eid'].apply(lambda x: x.item() if isinstance(x, torch.Tensor) else x)
        combined_csv_path = os.path.join(disease_output_dir, f"finetune_predictions_{disease_name}.csv")
        combined_df.to_csv(combined_csv_path, index=False)
        logging.info(f"[{disease_name}] Combined training and validation predictions saved at {combined_csv_path}")

    summary_metrics = {
        "Disease": disease_name,
        "Best_Epoch": best_epoch,
        "Val_AUC": best_val_auc,
        "Val_Acc": val_acc,
        "Val_Precision": val_precision,
        "Val_Recall": val_recall,
        "Val_F1": val_f1,
        "Train_Loss": np.mean(train_losses),
        "Val_Loss": np.mean(val_losses)
    }
    metrics_df = pd.DataFrame([summary_metrics])
    metrics_csv_path = os.path.join(disease_output_dir, f"finetune_metrics_{disease_name}.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    logging.info(f"[{disease_name}] Summary metrics saved at {metrics_csv_path}")

    overall_metrics = summary_metrics.copy()
    overall_metrics["Num_Samples"] = df_merged.shape[0]
    return overall_metrics

#################################
# 5. Main Program: Traverse Disease Label Files and Fine-Tune Sequentially
#################################
def main():
    summary_metrics_list = []
    label_files = glob.glob(os.path.join(labels_dir, "*_labels.csv"))
    logging.info(f"Found {len(label_files)} disease label files in total.")
    for file in label_files:
        metrics = train_finetune_for_disease(file)
        if metrics is not None:
            summary_metrics_list.append(metrics)

    if len(summary_metrics_list) > 0:
        summary_df = pd.DataFrame(summary_metrics_list)
        summary_csv_path = os.path.join(main_output_dir, "finetune_summary_metrics.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        logging.info(f"Summary of fine-tuning metrics for all diseases saved at {summary_csv_path}")
    else:
        logging.warning("No fine-tuning results for any disease!")

if __name__ == "__main__":
    main()