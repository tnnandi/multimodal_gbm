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

### 1. Modality-Specific Attention-Based MIL
- **Separate attention modules** for each modality (histology and spatial transcriptomics)
- Each module produces a patient-level embedding and an attention map for its modality
- **Weighted sum fusion**: The final patient embedding is `fused = alpha * histology_embedding + (1 - alpha) * st_embedding`, where `alpha` is user-specified (default 0.5)
- **Interpretability**: Both attention maps are available for analysis

### 2. Bayesian Framework
- **Epistemic Uncertainty**: Monte Carlo dropout during inference
- **Aleatoric Uncertainty**: Bootstrap sampling for confidence intervals
- **Survival Analysis**: Bayesian Cox proportional hazards or AFT models

## Schematic Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BAYESIAN MIL FOR GBM RECURRENCE PREDICTION              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PATIENT DATA  │    │   HISTOLOGY     │    │ SPATIAL RNA-SEQ │
│                 │    │   EMBEDDINGS    │    │   EMBEDDINGS    │
│  Patient_01     │    │  (2000 tiles)   │    │  (2000 tiles)   │
│  Patient_02     │    │  (512 dims)     │    │  (512 dims)     │
│  ...            │    │                 │    │                 │
│  Patient_17     │    └─────────────────┘    └─────────────────┘
└─────────────────┘              │                      │
                                 │                      │
                                 ▼                      ▼
                    ┌─────────────────────────────────────────┐
                    │   SEPARATE ATTENTION MIL MODULES        │
                    │   (one for each modality)               │
                    └─────────────────────────────────────────┘
                                 │                      │
                                 ▼                      ▼
                    ┌─────────────────────────────────────────┐
                    │   Patient-level Embedding (Histology)   │
                    └─────────────────────────────────────────┘
                    ┌─────────────────────────────────────────┐
                    │   Patient-level Embedding (ST)          │
                    └─────────────────────────────────────────┘
                                 │                      │
                                 └─────────┬────────────┘
                                           ▼
                    ┌─────────────────────────────────────────┐
                    │   Weighted Sum Fusion                   │
                    │   fused = alpha * hist + (1-alpha) * st │
                    └─────────────────────────────────────────┘
                                           │
                                           ▼
                    ┌─────────────────────────────────────────┐
                    │      CLASSIFIER / SURVIVAL HEAD         │
                    └─────────────────────────────────────────┘
                                           │
                                           ▼
                    ┌─────────────────────────────────────────┐
                    │         BAYESIAN SURVIVAL MODEL         │
                    └─────────────────────────────────────────┘
                                           │
                                           ▼
                    ┌─────────────────────────────────────────┐
                    │         UNCERTAINTY QUANTIFICATION      │
                    └─────────────────────────────────────────┘
                                           │
                                           ▼
                    ┌─────────────────────────────────────────┐
                    │              PREDICTIONS                │
                    └─────────────────────────────────────────┘
                                           │
                                           ▼
                    ┌─────────────────────────────────────────┐
                    │           INTERPRETABILITY              │
                    │   (Separate attention maps per modality)│
                    └─────────────────────────────────────────┘
```

## Key Features

- **Modality-Specific Attention**: Each modality has its own attention module and attention map
- **Weighted Fusion**: User controls the importance of each modality via `alpha`
- **Uncertainty Quantification**: Both epistemic and aleatoric uncertainty estimation
- **Interpretability**: Attention maps show which tiles contribute most to predictions, per modality
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
- Separate attention-based MIL modules for each modality
- Weighted sum fusion with user-specified `alpha`
- Bayesian survival analysis (Cox/AFT models)
- Bootstrap uncertainty estimation
- Train/test split only (no validation set)
- Comprehensive evaluation metrics

**Usage:**
```bash
# Training mode (with alpha=0.7 for 70% histology, 30% ST)
python train_model.bayesian.py --mode train --predict_type recurrence_time --survival_model cox --use_mc_dropout --alpha 0.7

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
- `--alpha`: Weight for histology modality in fusion (default: 0.5, range 0.0-1.0)

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
    def __init__(self, input_dim=512, hidden_dim=128, predict_type="recurrence_time", dropout_rate=0.2, alpha=0.5):
        # Separate AttentionMIL modules for each modality
        # Weighted sum fusion: fused = alpha * hist + (1-alpha) * st
        # Output classifier or survival head
```

**Key Components:**
1. **Separate Attention Modules**: Each modality has its own attention mechanism
2. **Weighted Fusion**: User controls the importance of each modality via `alpha`
3. **Classifier/Survival Head**: Output layer for prediction
4. **Interpretability**: Returns both attention maps

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
- **Attention Maps**: Tile-level importance visualization for each modality

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
    --alpha 0.7 \
    --plot_uncertainty
```

### 3. Train Model for Classification
```bash
python train_model.bayesian.py \
    --mode train \
    --predict_type recurrence_status \
    --use_mc_dropout \
    --alpha 0.5 \
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
- **Attention Maps**: Heatmaps showing tile importance for each modality
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



## Mathematical Formulation
                    
### Attention-Based MIL
                    
Given a bag of tiles $X = \{x_1, x_2, ..., x_T\}$ for a patient, where each tile $x_i \in \mathbb{R}^{512}$:
                    
1. **Embedding**: $h_i = \text{ReLU}(W_e x_i + b_e)$
2. **Attention**: $a_i = \frac{\exp(u^T \tanh(Vh_i))}{\sum_{j=1}^T \exp(u^T \tanh(Vh_j))}$
3. **Pooling**: $M = \sum_{i=1}^T a_i h_i$
4. **Fusion**: $M_\text{fused} = \alpha M_\text{hist} + (1-\alpha) M_\text{st}$
5. **Prediction**: $y = \sigma(W_c M_\text{fused} + b_c)$
                    
Where $W_e, V, U, W_c$ are learnable parameters and $\sigma$ is sigmoid activation.
                    
### Bayesian Uncertainty
                    
**Epistemic Uncertainty** (Model uncertainty):
- Monte Carlo dropout: $p(y|x) \approx \frac{1}{S} \sum_{s=1}^S p(y|x, \theta_s)$
- Variance: $\text{Var}[y] = \mathbb{E}[y^2] - (\mathbb{E}[y])^2$
                    
**Aleatoric Uncertainty** (Data uncertainty):
- Bootstrap sampling: $\text{CI}_{95\%} = [\text{percentile}_{2.5}, \text{percentile}_{97.5}]$
                    
### Survival Analysis
                    
**Cox Proportional Hazards**:
- Hazard function: $h(t|x) = h_0(t) \exp(\beta^T x)$
- Survival function: $S(t|x) = S_0(t)^{\exp(\beta^T x)}$
                    
**Accelerated Failure Time (AFT)**:
- Log-survival time: $\log(T) = \beta^T x + \sigma W$
- Where $W \sim \text{Weibull}(1, 1)$


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on the project repository or contact the development team.





