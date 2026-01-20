MetaboLM: A metabolomic language model for multi-disease early prediction and risk stratification
=====

As UKB has recently released the NMR metabolome data of approximately 500,000 people, we plan to update the model trained with more samples in the near future. ## 2026.1.20

## Instruction
MetaboLM is a transformer-based language model that was pre-trained on plasma metabolomics data from 83,744 healthy UK Biobank participants, enabling it to capture inherent patterns of metabolite interactions. We fine-tuned the model using metabolomics data from individuals diagnosed with 16 common chronic diseases. Additionally, MetaboLM can be adapted to support other custom fine-tuning tasks. 

![](https://github.com/Qiu-Shizheng/MetaboLM/blob/main/Figure%201.jpeg)

## System requirements
torch 2.4.1+cu124  
tqdm 4.66.4  
scikit-learn 1.4.2  
scipy 1.13.1  
seaborn 0.12.2  
python 3.11.9  
pytorch-cuda 12.4    
optuna 3.6.1     
numpy 1.26.4
matplotlib 3.8.4 

## Simulated dataset
Since the original data cannot be provided, we have provided a simulated dataset, simulated_metabolomics_data.csv, that has a similar distribution and the same metabolite composition as the original dataset.

## Weight
The weight for pre-training model is available at https://figshare.com/s/bd69f74785946802b585

The weight for fine-tuning models are available at 10.6084/m9.figshare.30744284
        
        
        
        

## DOI
https://doi.org/10.5281/zenodo.17083417
        
        
        
        
        
        
        
        
        
        
        
        
        
        
