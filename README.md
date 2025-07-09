# Bayesian Multi-Instance Learning (MIL) for Glioblastoma Recurrence Prediction

This project implements a Bayesian framework for predicting glioblastoma (GBM) recurrence using multimodal tile-level embeddings. The approach combines attention-based Multi-Instance Learning (MIL) with Bayesian survival analysis to provide both accurate predictions and uncertainty quantification.

## Overview

The model processes two modalities of tile-level embeddings:
- **Histology embeddings**: 512-dimensional features from tissue morphology
- **Spatial transcriptomics embeddings**: 512-dimensional features from gene expression patterns

Each patient has 2000 tiles per modality, and the model learns to:
1. Identify which tiles are most important for prediction (attention mechanism)
2. Aggregate tile-level features into patient-level representations
3. Predict recurrence status and time with uncertainty quantification

## Architecture

### 1. Attention-Based MIL
- **Input**: Tile embeddings (2000 tiles × 1024 dimensions per patient)
- **Attention Mechanism**: Learns importance weights for each tile
- **Output**: Patient-level embedding with interpretable attention maps

### 2. Bayesian Framework
- **Epistemic Uncertainty**: Monte Carlo dropout during inference
- **Aleatoric Uncertainty**: Bootstrap sampling for confidence intervals
- **Survival Analysis**: Bayesian Cox proportional hazards or AFT models

## Key Features

- **Multimodal Integration**: Combines histology and spatial transcriptomics data
- **Uncertainty Quantification**: Both epistemic and aleatoric uncertainty estimation
- **Interpretability**: Attention maps show which tiles contribute most to predictions
- **Survival Analysis**: Supports both classification (recurrence status) and survival (recurrence time) prediction
- **Bayesian Inference**: Full posterior distributions for model parameters

## Files

### `generate_synthetic_data.py`
Generates synthetic multimodal tile embeddings for testing and development.

**Features:**
- Creates realistic GBM recurrence data (70% recurrence rate, median ~8 months)
- Generates 2000 tiles per patient per modality
- Each tile has 512-dimensional embeddings
- Saves data in CSV format for easy loading
- Includes Kaplan-Meier survival curve visualization

**Usage:**
```bash
python generate_synthetic_data.py
```

**Output:**
- `synthetic_embeddings/` directory containing:
  - `patient_XX_histology.csv`: Histology embeddings for each patient
  - `patient_XX_spatial_rnaseq.csv`: Spatial transcriptomics embeddings
  - `labels.csv`: Patient-level recurrence labels
  - `synthetic_km_curve.png`: Survival curve visualization

### `train_model.bayesian.py`
Main training script for the Bayesian MIL model.

**Features:**
- Attention-based MIL with Monte Carlo dropout
- Bayesian survival analysis (Cox/AFT models)
- Bootstrap uncertainty estimation
- Cross-validation support
- Comprehensive evaluation metrics

**Usage:**
```bash
# Training mode
python train_model.bayesian.py --mode train --predict_type recurrence_time --survival_model cox --use_mc_dropout

# Testing mode
python train_model.bayesian.py --mode test --predict_type recurrence_status
```

## Installation

### Requirements
```bash
pip install torch torchvision
pip install pandas numpy matplotlib seaborn
pip install scikit-learn lifelines
pip install pymc arviz
```

### Data Structure
```
synthetic_embeddings/
├── patient_01_histology.csv
├── patient_01_spatial_rnaseq.csv
├── patient_02_histology.csv
├── patient_02_spatial_rnaseq.csv
├── ...
├── labels.csv
└── synthetic_km_curve.png
```

## Model Parameters

### Attention MIL Parameters
- `--hidden_dim`: Hidden dimension for attention mechanism (default: 128)
- `--dropout_rate`: Dropout rate for Monte Carlo dropout (default: 0.2)
- `--learning_rate`: Learning rate for optimization (default: 1e-3)
- `--weight_decay`: Weight decay for regularization (default: 1e-4)

### Training Parameters
- `--num_epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 1 for MIL)
- `--n_folds`: Number of cross-validation folds (default: 5)

### Uncertainty Parameters
- `--use_mc_dropout`: Enable Monte Carlo dropout for epistemic uncertainty
- `--n_mc_samples`: Number of MC samples (default: 100)
- `--do_bootstrap`: Enable bootstrap sampling for aleatoric uncertainty
- `--n_bootstraps`: Number of bootstrap samples (default: 100)

### Prediction Parameters
- `--predict_type`: Type of prediction (`recurrence_status` or `recurrence_time`)
- `--survival_model`: Survival model type (`cox` or `aft`)

## Model Architecture Details

### BayesianAttentionMIL Class
```python
class BayesianAttentionMIL(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=128, 
                 predict_type="recurrence_time", dropout_rate=0.2):
        # Embedding layer with dropout
        # Attention mechanism (V, U matrices)
        # Output classifier with dropout
```

**Key Components:**
1. **Embedding Layer**: Projects tile embeddings to latent space
2. **Attention Mechanism**: 
   - `attention_V`: Intermediate transformation
   - `attention_U`: Linear projection to attention scores
   - Softmax normalization for attention weights
3. **Pooling**: Weighted aggregation of tile features
4. **Classifier**: Output layer with dropout for uncertainty

### Bayesian Survival Model
```python
class BayesianSurvivalModel:
    def __init__(self, survival_model='cox'):
        # Supports Cox proportional hazards and AFT models
```

**Features:**
- **Cox Model**: Non-parametric baseline hazard with parametric covariates
- **AFT Model**: Weibull distribution for survival times
- **PyMC Integration**: Full Bayesian inference with MCMC sampling

## Evaluation Metrics

### Classification (Recurrence Status)
- **Accuracy**: Overall classification accuracy
- **ROC-AUC**: Area under ROC curve
- **Uncertainty**: Monte Carlo dropout variance

### Survival Analysis (Recurrence Time)
- **Concordance Index**: Harrell's C-index for survival discrimination
- **Survival Curves**: Kaplan-Meier plots with uncertainty bands
- **Risk Scores**: Individual patient risk predictions

### Uncertainty Quantification
- **Epistemic Uncertainty**: Model uncertainty (MC dropout variance)
- **Aleatoric Uncertainty**: Data uncertainty (bootstrap confidence intervals)
- **Attention Maps**: Tile-level importance visualization

## Example Usage

### 1. Generate Synthetic Data
```bash
python generate_synthetic_data.py
```

### 2. Train Model for Survival Prediction
```bash
python train_model.bayesian.py \
    --mode train \
    --predict_type recurrence_time \
    --survival_model cox \
    --use_mc_dropout \
    --n_mc_samples 100 \
    --do_bootstrap \
    --n_bootstraps 100 \
    --plot_uncertainty
```

### 3. Train Model for Classification
```bash
python train_model.bayesian.py \
    --mode train \
    --predict_type recurrence_status \
    --use_mc_dropout \
    --plot_attention
```

### 4. Test Trained Model
```bash
python train_model.bayesian.py \
    --mode test \
    --predict_type recurrence_time
```

## Output Files

### Training Outputs
- `models/mil_bayesian_model.pt`: Saved model state
- `uncertainty_analysis_YYYY-MM-DD_HH-MM-SS.png`: Uncertainty visualization
- Console output with training metrics and validation scores

### Visualization Features
- **Attention Maps**: Heatmaps showing tile importance
- **Uncertainty Analysis**: Predictions vs uncertainty scatter plots
- **Survival Curves**: Kaplan-Meier plots with confidence bands
- **ROC Curves**: For classification tasks

## Technical Notes

### Data Format
- **Tile Embeddings**: CSV files with columns `tile_id, dim_1, dim_2, ..., dim_512`
- **Labels**: CSV with columns `patient_id, recurrence_status, recurrence_time`
- **Patient IDs**: Format `patient_XX` where XX is zero-padded patient number

### Memory Considerations
- Each patient requires ~8MB for tile embeddings (2000 tiles × 1024 dims × 4 bytes)
- Monte Carlo sampling increases memory usage by factor of `n_mc_samples`
- Bootstrap sampling requires multiple model evaluations

<!-- ### Performance Tips
- Use GPU acceleration when available (`torch.cuda.is_available()`)
- Reduce `n_mc_samples` for faster inference
- Use smaller `n_bootstraps` for quicker uncertainty estimation
- Enable `torch.backends.cudnn.benchmark = True` for faster GPU operations -->

## Future Enhancements

1. **Co-Attention Mechanism/Cross-Modal Contrastive Learner**: Learn cross-modal attention patterns


