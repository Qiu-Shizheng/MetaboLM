import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import warnings
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertConfig, get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import seaborn as sns
from datetime import datetime

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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

# Create a new folder to save all results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("your/path", f"metabolite_pretraining_{timestamp}")
os.makedirs(output_dir, exist_ok=True)
logging.info(f"All results will be saved in directory: {output_dir}")

# ===== Data Loading and Preprocessing =====
# 1. Load healthy individuals' eid list
logging.info("Loading healthy individuals' eid list...")
healthy_eids_path = "your/path/healthy_eids.csv"
healthy_eids = pd.read_csv(healthy_eids_path)["eid"].tolist()

# 2. Load metabolite expression data for all participants
logging.info("Loading metabolite expression data for all participants...")
expression_data_path = "your/path/metabolomic.csv"
expression_data = pd.read_csv(expression_data_path)

# 3. Filter data for healthy individuals and set eid as index
logging.info("Filtering healthy individuals' data...")
expression_data = expression_data[expression_data["eid"].isin(healthy_eids)]
expression_data.set_index("eid", inplace=True)

num_participants = expression_data.shape[0]
num_metabolites_all = len(expression_data.columns)
logging.info(f"Number of participants in expression data: {num_participants}")
logging.info(f"Initial number of metabolites: {num_metabolites_all}")

# 4. Load metabolite correlation matrix
logging.info("Loading metabolite correlation matrix...")
correlation_matrix_path = "your/path/metabolomic_correlation_matrix.csv"
correlation_matrix = pd.read_csv(correlation_matrix_path, index_col=0)
logging.info(f"Original shape of correlation matrix: {correlation_matrix.shape}")

# 5. Take the intersection of metabolites from expression data and correlation matrix
common_metabolites = sorted(
    list(
        set(expression_data.columns)
        & set(correlation_matrix.index)
        & set(correlation_matrix.columns)
    )
)

expression_data = expression_data[common_metabolites]
correlation_matrix = correlation_matrix.loc[common_metabolites, common_metabolites]
logging.info(f"Number of metabolites after filtering in expression data: {expression_data.shape[1]}")
logging.info(f"Shape of correlation matrix after filtering: {correlation_matrix.shape}")

# 6. Convert the correlation matrix to a tensor, and pad 0 for the [CLS] token (in the first row)
bias_matrix = torch.tensor(correlation_matrix.values, dtype=torch.float32, device=device)
logging.info(f"Tensor shape of correlation matrix (without CLS token): {bias_matrix.shape}")
bias_matrix = F.pad(bias_matrix, (1, 0, 1, 0), "constant", 0)
logging.info(f"Shape of correlation matrix with CLS token considered: {bias_matrix.shape}")

# 7. Split the data into training and validation sets (80%/20%)
logging.info("Splitting metabolite data into training and validation sets...")
train_expression_data = expression_data.sample(frac=0.8, random_state=42)
val_expression_data = expression_data.drop(train_expression_data.index)
logging.info(f"Number of participants in training set: {train_expression_data.shape[0]}, Validation set: {val_expression_data.shape[0]}")

# 8. Data normalization: using z-score
train_mean = train_expression_data.mean(axis=0)
train_std = train_expression_data.std(axis=0)
train_std[train_std == 0] = 1.0
train_expression_data = (train_expression_data - train_mean) / train_std
val_expression_data = (val_expression_data - train_mean) / train_std
logging.info("Normalization complete: used training set mean and standard deviation for z-score transformation.")

# Update the range for random replacement values
random_min = train_expression_data.min().min()
random_max = train_expression_data.max().max()
logging.info(f"Updated range for random replacement values: min={random_min}, max={random_max}")


# ===== Dataset Definition =====
class MetaboliteExpressionDataset(Dataset):
    def __init__(self, expression_data):
        self.expression_data = expression_data.values.astype(np.float32)
        self.num_samples = expression_data.shape[0]
        self.num_metabolites = expression_data.shape[1]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        expressions = self.expression_data[idx]
        return torch.tensor(expressions, dtype=torch.float32)


train_dataset = MetaboliteExpressionDataset(train_expression_data)
val_dataset = MetaboliteExpressionDataset(val_expression_data)

batch_size = 512
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

# ===== Custom BERT Model and Attention Mechanism =====
# - Because metabolomics data does not have an inherent order, positional encoding is not added;
# - Include the correlation matrix bias in the attention mechanism;

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
        bias_coef=None,
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
            bias_coef=bias_coef,
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
            bias_coef=bias_coef,
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
                bias_coef=bias_coef,
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
                bias_coef=bias_coef,
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
                v
                for v in [hidden_states, next_decoder_cache, all_hidden_states, all_attentions]
                if v is not None
            )
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_decoder_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
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
            token_type_ids = torch.zeros(
                input_shape,
                dtype=torch.long,
                device=inputs_embeds.device if inputs_embeds is not None else device,
            )

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length),
                device=inputs_embeds.device if inputs_embeds is not None else device,
            )

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)

        if encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    (encoder_batch_size, encoder_sequence_length),
                    device=inputs_embeds.device if inputs_embeds is not None else device,
                )
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

        sequence_output = encoder_outputs["last_hidden_state"]
        pooled_output = (
            self.pooler(sequence_output.transpose(1, 2)).squeeze(-1) if self.pooler is not None else None
        )

        if not return_dict:
            return (
                sequence_output,
                pooled_output,
            ) + tuple(
                v
                for v in [encoder_outputs["past_key_values"], encoder_outputs["hidden_states"], encoder_outputs["attentions"]]
                if v is not None
            )

        return {
            "last_hidden_state": sequence_output,
            "pooler_output": pooled_output,
            "past_key_values": encoder_outputs["past_key_values"],
            "hidden_states": encoder_outputs["hidden_states"],
            "attentions": encoder_outputs["attentions"],
        }


# ===== Define BERT Model =====
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
        # Use element-wise (Hadamard) mapping: assign each metabolite a learnable weight vector and bias
        self.expr_weight = nn.Parameter(torch.ones(1, num_metabolites, hidden_size))
        self.expr_bias = nn.Parameter(torch.zeros(1, num_metabolites, hidden_size))
        # [CLS] token, used for global information summarization
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.output_layer = nn.Linear(hidden_size, 1)
        self.register_buffer("bias_matrix_full", bias_matrix)
        self.bias_coef = nn.Parameter(torch.tensor(0.01))

    def forward(self, expressions, attention_mask):
        batch_size, num_metabolites = expressions.size()
        # Use element-wise mapping: first expand dimension (each metabolite expression is a scalar), then multiply element-wise with learnable weights and add bias
        expr_embeds = expressions.unsqueeze(-1) * self.expr_weight + self.expr_bias  # [B, num_metabolites, hidden_size]
        cls_tokens = self.cls_token.expand(batch_size, 1, -1)  # [B, 1, hidden_size]
        embeddings = torch.cat((cls_tokens, expr_embeds), dim=1)  # [B, num_metabolites+1, hidden_size]
        new_attention_mask = torch.cat((torch.ones(batch_size, 1, device=attention_mask.device), attention_mask), dim=1)
        outputs = self.bert(
            inputs_embeds=embeddings,
            attention_mask=new_attention_mask,
            output_attentions=True,
            bias_matrix_chunk=self.bias_matrix_full,
            bias_coef=self.bias_coef,
        )

        prediction_scores = self.output_layer(outputs["last_hidden_state"][:, 1:, :]).squeeze(-1)
        attentions = outputs["attentions"]
        return prediction_scores, attentions


# ===== Hyperparameters =====
hidden_size = 768
num_layers = 12
learning_rate = 5e-5
num_epochs = 100

model = MetaboliteBERTModel(hidden_size, num_layers, expression_data.shape[1], bias_matrix)
if gpu_count > 1:
    model = nn.DataParallel(model)
model.to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
total_steps = len(train_dataloader) * num_epochs
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
)

tolerance_value = 0.5

# ===== Record Epoch Metrics for Training and Validation =====
train_loss_values = []
train_acc_values = []
train_mse_values = []
train_mae_values = []
train_r2_values = []

val_loss_values = []
val_acc_values = []
val_mse_values = []
val_mae_values = []
val_r2_values = []

best_val_loss = float("inf")
best_epoch = -1

logging.info("Starting pretraining of Metabolite BERT model...")
for epoch in range(num_epochs):
    logging.info(f"Epoch {epoch + 1}/{num_epochs} training started...")
    model.train()
    epoch_loss = 0.0
    epoch_total_correct = 0
    epoch_total_pred = 0
    epoch_sum_squared_errors = 0.0
    epoch_sum_abs_errors = 0.0
    epoch_sum_actual = 0.0
    epoch_sum_actual_sq = 0.0
    epoch_masked_count = 0

    train_progress = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
    for expressions in train_progress:
        expressions = expressions.to(device)
        attention_mask = torch.ones(expressions.size(), dtype=torch.long, device=device)

        # Generate mask: mask 15% positions
        probability_matrix = torch.full(expressions.size(), 0.15, device=device)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        if masked_indices.sum().item() == 0:
            continue

        replace_prob = torch.rand(masked_indices.sum(), device=device)
        mask_as_zero = replace_prob < 0.8
        mask_as_random = (replace_prob >= 0.8) & (replace_prob < 0.9)

        labels = expressions.clone()
        expressions_masked = expressions.clone()
        # Replace with 0 in 80% of the masked positions
        expressions_masked[masked_indices] = 0.0

        if mask_as_random.sum() > 0:
            random_values = torch.empty(mask_as_random.sum(), device=device).uniform_(random_min, random_max)
            temp = expressions_masked[masked_indices]
            temp[mask_as_random] = random_values
            expressions_masked[masked_indices] = temp

        num_masked = masked_indices.sum().item()
        optimizer.zero_grad()
        outputs, _ = model(expressions_masked, attention_mask)
        loss = criterion(outputs[masked_indices], labels[masked_indices])
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()

        predicted_values = outputs[masked_indices]
        actual_values = labels[masked_indices]
        correct = (torch.abs(predicted_values - actual_values) < tolerance_value).sum().item()
        epoch_total_correct += correct
        epoch_total_pred += num_masked

        errors = predicted_values - actual_values
        epoch_sum_squared_errors += (errors ** 2).sum().item()
        epoch_sum_abs_errors += errors.abs().sum().item()
        epoch_sum_actual += actual_values.sum().item()
        epoch_sum_actual_sq += (actual_values ** 2).sum().item()
        epoch_masked_count += num_masked

        train_progress.set_postfix({"Loss": loss.item()})

    avg_train_loss = epoch_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
    train_loss_values.append(avg_train_loss)
    train_acc = epoch_total_correct / epoch_total_pred if epoch_total_pred > 0 else 0
    train_acc_values.append(train_acc)
    if epoch_masked_count > 0:
        train_mse = epoch_sum_squared_errors / epoch_masked_count
        train_mae = epoch_sum_abs_errors / epoch_masked_count
        sst = epoch_sum_actual_sq - (epoch_sum_actual ** 2 / epoch_masked_count)
        train_r2 = 1 - (epoch_sum_squared_errors / sst) if sst > 0 else 0.0
    else:
        train_mse, train_mae, train_r2 = 0.0, 0.0, 0.0
    train_mse_values.append(train_mse)
    train_mae_values.append(train_mae)
    train_r2_values.append(train_r2)

    # ===== Validation Phase =====
    model.eval()
    val_epoch_loss = 0.0
    val_total_correct = 0
    val_total_pred = 0
    val_sum_squared_errors = 0.0
    val_sum_abs_errors = 0.0
    val_sum_actual = 0.0
    val_sum_actual_sq = 0.0
    val_masked_count = 0
    with torch.no_grad():
        for expressions in val_dataloader:
            expressions = expressions.to(device)
            attention_mask = torch.ones(expressions.size(), dtype=torch.long, device=device)

            # Generate mask: mask 15% positions
            probability_matrix = torch.full(expressions.size(), 0.15, device=device)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            if masked_indices.sum().item() == 0:
                continue

            replace_prob = torch.rand(masked_indices.sum(), device=device)
            mask_as_zero = replace_prob < 0.8
            mask_as_random = (replace_prob >= 0.8) & (replace_prob < 0.9)

            labels = expressions.clone()
            expressions_masked = expressions.clone()
            # Replace with 0 in 80% of the masked positions
            expressions_masked[masked_indices] = 0.0

            if mask_as_random.sum() > 0:
                random_values = torch.empty(mask_as_random.sum(), device=device).uniform_(random_min, random_max)
                temp = expressions_masked[masked_indices]
                temp[mask_as_random] = random_values
                expressions_masked[masked_indices] = temp
            num_masked = masked_indices.sum().item()
            outputs, _ = model(expressions_masked, attention_mask)
            loss = criterion(outputs[masked_indices], labels[masked_indices])
            val_epoch_loss += loss.item()

            predicted_values = outputs[masked_indices]
            actual_values = labels[masked_indices]
            correct = (torch.abs(predicted_values - actual_values) < tolerance_value).sum().item()
            val_total_correct += correct
            val_total_pred += num_masked

            errors = predicted_values - actual_values
            val_sum_squared_errors += (errors ** 2).sum().item()
            val_sum_abs_errors += errors.abs().sum().item()
            val_sum_actual += actual_values.sum().item()
            val_sum_actual_sq += (actual_values ** 2).sum().item()
            val_masked_count += num_masked

    avg_val_loss = val_epoch_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0
    val_loss_values.append(avg_val_loss)
    val_acc = val_total_correct / val_total_pred if val_total_pred > 0 else 0
    val_acc_values.append(val_acc)
    if val_masked_count > 0:
        val_mse = val_sum_squared_errors / val_masked_count
        val_mae = val_sum_abs_errors / val_masked_count
        sst = val_sum_actual_sq - (val_sum_actual ** 2 / val_masked_count)
        val_r2 = 1 - (val_sum_squared_errors / sst) if sst > 0 else 0.0
    else:
        val_mse, val_mae, val_r2 = 0.0, 0.0, 0.0
    val_mse_values.append(val_mse)
    val_mae_values.append(val_mae)
    val_r2_values.append(val_r2)

    logging.info(
        f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Train MSE: {train_mse:.4f}, Train MAE: {train_mae:.4f}, Train R²: {train_r2:.4f}"
    )
    logging.info(
        f"Epoch {epoch + 1}: Val   Loss: {avg_val_loss:.4f}, Val   Acc: {val_acc:.4f}, Val   MSE: {val_mse:.4f}, Val   MAE: {val_mae:.4f}, Val   R²: {val_r2:.4f}"
    )

    # Save the model with the best performance on the validation set
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
        model_to_save = model.module if isinstance(model, nn.DataParallel) else model
        model_path = os.path.join(output_dir, "best_metabolite_bert_model.pt")
        torch.save(model_to_save.state_dict(), model_path)
        logging.info(f"Epoch {epoch + 1}: New best validation loss {best_val_loss:.4f}, model saved at {model_path}")

# ===== Plotting and Metrics Report =====
# Plot training and validation loss curves
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_loss_values, marker="o", label="Train Loss")
plt.plot(range(1, num_epochs + 1), val_loss_values, marker="o", label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
loss_plot_path = os.path.join(output_dir, "pretraining_loss.pdf")
plt.savefig(loss_plot_path)
plt.show()

# Plot Mask Prediction Accuracy
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_acc_values, marker="o", label="Train Mask Acc")
plt.plot(range(1, num_epochs + 1), val_acc_values, marker="o", label="Val Mask Acc")
plt.xlabel("Epoch")
plt.ylabel("Mask Prediction Accuracy")
plt.title("Mask Prediction Accuracy vs Epoch")
plt.legend()
plt.grid(True)
acc_plot_path = os.path.join(output_dir, "mask_accuracy.pdf")
plt.savefig(acc_plot_path)
plt.show()

# Plot MSE curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_mse_values, marker="o", label="Train MSE")
plt.plot(range(1, num_epochs + 1), val_mse_values, marker="o", label="Val MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("MSE vs Epoch")
plt.legend()
plt.grid(True)
mse_plot_path = os.path.join(output_dir, "mse.pdf")
plt.savefig(mse_plot_path)
plt.show()

# Plot MAE curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_mae_values, marker="o", label="Train MAE")
plt.plot(range(1, num_epochs + 1), val_mae_values, marker="o", label="Val MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("MAE vs Epoch")
plt.legend()
plt.grid(True)
mae_plot_path = os.path.join(output_dir, "mae.pdf")
plt.savefig(mae_plot_path)
plt.show()

# Plot R² curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_r2_values, marker="o", label="Train R²")
plt.plot(range(1, num_epochs + 1), val_r2_values, marker="o", label="Val R²")
plt.xlabel("Epoch")
plt.ylabel("R²")
plt.title("R² vs Epoch")
plt.legend()
plt.grid(True)
r2_plot_path = os.path.join(output_dir, "r2.pdf")
plt.savefig(r2_plot_path)
plt.show()

logging.info("====== Best Validation Metrics ======")
logging.info(f"Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch}")
logging.info(f"Best Val Mask Accuracy: {max(val_acc_values):.4f}")
logging.info(f"Best Val MSE: {min(val_mse_values):.4f}")
logging.info(f"Best Val MAE: {min(val_mae_values):.4f}")
logging.info(f"Best Val R²: {max(val_r2_values):.4f}")

# Save all metrics to a CSV file
metrics_df = pd.DataFrame({
    "Epoch": range(1, num_epochs + 1),
    "Train_Loss": train_loss_values,
    "Val_Loss": val_loss_values,
    "Train_Acc": train_acc_values,
    "Val_Acc": val_acc_values,
    "Train_MSE": train_mse_values,
    "Val_MSE": val_mse_values,
    "Train_MAE": train_mae_values,
    "Val_MAE": val_mae_values,
    "Train_R2": train_r2_values,
    "Val_R2": val_r2_values,
})
metrics_csv_path = os.path.join(output_dir, "training_metrics.csv")
metrics_df.to_csv(metrics_csv_path, index=False)
logging.info(f"All metrics have been saved to {metrics_csv_path}")

# ===== Interpretability Analysis =====
logging.info("Starting interpretability analysis: extracting attention weights for all metabolites...")
model.eval()
sum_attention_weights = torch.zeros(expression_data.shape[1], expression_data.shape[1])
total_samples = 0

with torch.no_grad():
    for expressions in tqdm(val_dataloader, desc="Extracting attention weights"):
        expressions = expressions.to(device)
        attention_mask = torch.ones(expressions.size(), dtype=torch.long, device=device)
        outputs, attentions = model(expressions, attention_mask)
        # Take the attention from the last layer and average across all heads
        att = attentions[-1]  # [B, num_heads, seq_len, seq_len]
        att = att.mean(dim=1)  # [B, seq_len, seq_len]
        # Remove the [CLS] token corresponding row and column
        att = att[:, 1:, 1:]
        sum_attention_weights += att.sum(dim=0).cpu()
        total_samples += expressions.size(0)

mean_attention_weights = sum_attention_weights / total_samples
metabolite_importance = mean_attention_weights.sum(dim=0).numpy()
metabolite_names = common_metabolites

importance_df = pd.DataFrame({
    "Metabolite": metabolite_names,
    "Attention_Weight": metabolite_importance
})
importance_df = importance_df.sort_values(by="Attention_Weight", ascending=False)
importance_csv_path = os.path.join(output_dir, "metabolite_importance.csv")
importance_df.to_csv(importance_csv_path, index=False)
logging.info(f"All metabolites' attention weight results have been saved to {importance_csv_path}")
logging.info("Top 10 important metabolites:")
logging.info(f"\n{importance_df.head(10)}")

plt.figure(figsize=(12, 7))
sns.barplot(x="Attention_Weight", y="Metabolite", data=importance_df.head(10))
plt.title("Top 10 Important Metabolites by Attention Weights")
plt.xlabel("Attention Weight")
plt.ylabel("Metabolite")
plt.tight_layout()
importance_plot_path = os.path.join(output_dir, "important_metabolites.pdf")
plt.savefig(importance_plot_path)
plt.show()