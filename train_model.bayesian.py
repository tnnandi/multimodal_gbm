# Multimodal MIL model for GBM recurrence prediction using synthetic tile embeddings
# num_patients = 17, num_tiles = 2000, embedding_dim = 512
# total number of tiles = 17 * 2000 = 34000

# Steps involving MIL:
# 1. Learn which tiles are most important for predicting the bag-level label
# 2. Aggregate the tile embeddings into a single, meaningful patient-level embedding
# 3. Use the patient-level embedding to predict the bag-level label

# Bayesian Framework:
# 1. Attention-based MIL pooling to produce interpretable attention maps over tiles
# 2. Bayesian Cox or AFT model on the pooled patient embedding for uncertainty quantification
# 3. Monte Carlo dropout for epistemic uncertainty
# 4. Bootstrap sampling for aleatoric uncertainty

# Key components:
# Attention based MIL pooling:
#   1. BayesianAttentionMIL class with Monte Carlo dropout for epistemic uncertainty
#   2. Attention mechanism to learn which tiles are most important
#   3. Monte Carlo dropout for epistemic uncertainty

# Bayesian survival model:


# To do:
# 1. add co-attention mechanism to learn which tiles are most important for each other


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter, WeibullAFTFitter
from lifelines.utils import concordance_index
import pymc as pm
import arviz as az
from scipy import stats
import seaborn as sns
import multiprocessing
import random
import datetime
import argparse
import pickle
from pdb import set_trace

# Set random seeds for reproducibility
seed_value = 142  
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

# Ensures reproducible (but potentially slower) behavior on GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def parse_args():
    parser = argparse.ArgumentParser(description='Bayesian MIL model training and evaluation parameters')
    
    # Model and data parameters
    parser.add_argument('--predict_type', type=str, default='recurrence_time', 
                        choices=['recurrence_status', 'recurrence_time'],
                        help='Type of prediction: recurrence_status (classification) or recurrence_time (survival)')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'test'],
                        help='Mode: train or test')
    parser.add_argument('--survival_model', type=str, default='cox', 
                        choices=['cox', 'aft'],
                        help='Survival model type: cox (proportional hazards) or aft (accelerated failure time)')
    parser.add_argument('--use_mc_dropout', action='store_true', default=False,
                        help='Use Monte Carlo dropout for epistemic uncertainty')
    parser.add_argument('--n_mc_samples', type=int, default=100,
                        help='Number of Monte Carlo samples for uncertainty estimation')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension for attention mechanism')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout rate for Monte Carlo dropout')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (1 for MIL)')
    
    # Evaluation parameters
    parser.add_argument('--do_bootstrap', action='store_true', default=True,
                        help='Perform bootstrap resampling for confidence intervals')
    parser.add_argument('--n_bootstraps', type=int, default=100,
                        help='Number of bootstrap samples')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds for cross-validation')
    parser.add_argument('--plot_uncertainty', action='store_true',
                        help='Plot uncertainty estimates')
    parser.add_argument('--plot_attention', action='store_true',
                        help='Plot attention weights for interpretability')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./synthetic_embeddings',
                        help='Directory containing the synthetic embeddings')
    parser.add_argument('--model_save_path', type=str, default='./models/mil_bayesian_model.pt',
                        help='Path to save the trained model')
    
    return parser.parse_args()

args = parse_args()

# Configuration
DATA_DIR = args.data_dir
MODEL_SAVE_PATH = args.model_save_path
predict_type = args.predict_type
mode = args.mode
survival_model = args.survival_model
use_mc_dropout = args.use_mc_dropout
n_mc_samples = args.n_mc_samples
num_epochs = args.num_epochs

# Load labels
labels_df = pd.read_csv(os.path.join(DATA_DIR, 'labels.csv'))
labels_df.set_index('patient_id', inplace=True)

# Dataset class
class MultimodalMILBag(Dataset):
    def __init__(self, labels_df, data_dir, predict_type="recurrence_status"):
        super().__init__()
        self.labels_df = labels_df # slide-level labels
        self.data_dir = data_dir
        self.predict_type = predict_type
        self.patients = labels_df.index.tolist()

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_id = self.patients[idx]

        # Load histology embeddings
        hist_file = os.path.join(self.data_dir, f"{patient_id}_histology.csv")
        hist_df = pd.read_csv(hist_file)
        hist_embeddings = hist_df.drop(columns=['tile_id']).values  # shape: (num_tiles, embedding_dim)
        hist_tensor = torch.tensor(hist_embeddings, dtype=torch.float32)

        # Load spatial transcriptomics embeddings
        st_file = os.path.join(self.data_dir, f"{patient_id}_spatial_rnaseq.csv")
        st_df = pd.read_csv(st_file)
        st_embeddings = st_df.drop(columns=['tile_id']).values  # shape: (num_tiles, embedding_dim)
        st_tensor = torch.tensor(st_embeddings, dtype=torch.float32)

        # Concatenate modalities for each tile
        # shape: (num_tiles, 2*embedding_dim)
        # TODO: consider other ways to combine modalities (e.g., weighted sum) to keep the total embedding dim the same
        combined_tiles = torch.cat([hist_tensor, st_tensor], dim=1)

        # Load label
        if self.predict_type == "recurrence_status":
            label = self.labels_df.loc[patient_id, 'recurrence_status']
            time = torch.tensor(0.0, dtype=torch.float32)  # Not used for classification
            event = torch.tensor(label, dtype=torch.float32)
        elif self.predict_type == "recurrence_time":
            time = self.labels_df.loc[patient_id, 'recurrence_time']
            event = self.labels_df.loc[patient_id, 'recurrence_status']
            time_tensor = torch.tensor(time, dtype=torch.float32)
            event_tensor = torch.tensor(event, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported predict_type: {self.predict_type}")

        if self.predict_type == "recurrence_status":
            return combined_tiles, event
        else:
            return combined_tiles, time_tensor, event_tensor

# Bayesian MIL model with attention pooling and Monte Carlo dropout
class BayesianAttentionMIL(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=128, predict_type="recurrence_time", dropout_rate=0.2):
        super(BayesianAttentionMIL, self).__init__()
        self.predict_type = predict_type
        self.dropout_rate = dropout_rate
        
        # Embedding layers with dropout for epistemic uncertainty
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Attention mechanism
        self.attention_V = nn.Linear(hidden_dim, hidden_dim)
        self.attention_U = nn.Linear(hidden_dim, 1)
        
        # Output layer with dropout
        if predict_type == "recurrence_status":
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, 1)
            )
        else:
            # For survival analysis, output is risk score
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, x, training=True):
        # Apply dropout during training and MC sampling
        if training:
            self.train()
        else:
            self.eval()
        
        H = self.embedding(x)  # [T, H] # Project tile embedding to latent space
        A = torch.tanh(self.attention_V(H)) # Intermediate attention transformation
        A = self.attention_U(A) # Linear projection to scalar attention score
        A = torch.softmax(A, dim=0)  # normalized attention weights # Ilse 2018, Eq 8
        M = torch.sum(A * H, dim=0)  # [H] Aggregate to patient-level embedding # Ilse 2018, Eq 7
        output = self.classifier(M)  # [1] # Predict risk or label from pooled representation
        return output.squeeze(), A

    # can turn it off for now
    def mc_predict(self, x, n_samples=100):
        """Monte Carlo prediction for uncertainty estimation"""
        predictions = []
        attention_maps = []
        
        for _ in range(n_samples):
            with torch.no_grad():
                pred, attn = self.forward(x, training=True)  # Keep dropout active
                predictions.append(pred.item())
                attention_maps.append(attn.cpu().numpy())
        
        predictions = np.array(predictions)
        attention_maps = np.array(attention_maps)
        
        return predictions, attention_maps

# Cox loss for survival analysis
def cox_loss(risks, times, events):
    order = torch.argsort(times, descending=True)
    risks = risks[order]
    events = events[order]
    log_cumsum_exp = torch.logcumsumexp(risks, dim=0)
    loss = -torch.sum((risks - log_cumsum_exp) * events) / events.sum()
    return loss

# Binary cross entropy loss for classification
def bce_loss(predictions, targets):
    return F.binary_cross_entropy_with_logits(predictions, targets)

# Bayesian survival model using PyMC
class BayesianSurvivalModel:
    def __init__(self, survival_model='cox'):
        self.survival_model = survival_model
        self.model = None
        self.trace = None
        
    def fit(self, X, times, events):
        """Fit Bayesian survival model"""
        print(f"Fitting Bayesian {self.survival_model.upper()} model...")
        
        if self.survival_model == 'cox':
            self._fit_cox_model(X, times, events)
        elif self.survival_model == 'aft':
            self._fit_aft_model(X, times, events)
        else:
            raise ValueError(f"Unsupported survival model: {self.survival_model}")
    
    def _fit_cox_model(self, X, times, events):
        """Fit Bayesian Cox proportional hazards model"""
        n_features = X.shape[1]
        
        with pm.Model() as self.model:
            # Priors for coefficients
            beta = pm.Normal('beta', mu=0, sigma=1, shape=n_features)
            
            # Baseline hazard (non-parametric)
            baseline_hazard = pm.Gamma('baseline_hazard', alpha=1, beta=1, shape=len(np.unique(times)))
            
            # Linear predictor
            linear_predictor = pm.math.dot(X, beta)
            
            # Likelihood
            for i in range(len(times)):
                if events[i] == 1:  # Event occurred
                    pm.Exponential(f'likelihood_{i}', 
                                 lam=baseline_hazard[times[i]] * pm.math.exp(linear_predictor[i]),
                                 observed=1)
                else:  # Censored
                    pm.Exponential(f'likelihood_{i}', 
                                 lam=baseline_hazard[times[i]] * pm.math.exp(linear_predictor[i]),
                                 observed=0)
            
            # Sample from posterior
            self.trace = pm.sample(1000, tune=1000, return_inferencedata=True)
    
    def _fit_aft_model(self, X, times, events):
        """Fit Bayesian Accelerated Failure Time model"""
        n_features = X.shape[1]
        
        with pm.Model() as self.model:
            # Priors for coefficients
            beta = pm.Normal('beta', mu=0, sigma=1, shape=n_features)
            sigma = pm.HalfNormal('sigma', sigma=1)
            
            # Linear predictor
            linear_predictor = pm.math.dot(X, beta)
            
            # Likelihood (Weibull distribution)
            for i in range(len(times)):
                if events[i] == 1:  # Event occurred
                    pm.Weibull(f'likelihood_{i}', 
                              alpha=1/sigma, 
                              beta=pm.math.exp(-linear_predictor[i]/sigma),
                              observed=times[i])
                else:  # Censored
                    pm.Weibull(f'likelihood_{i}', 
                              alpha=1/sigma, 
                              beta=pm.math.exp(-linear_predictor[i]/sigma),
                              observed=times[i])
            
            # Sample from posterior
            self.trace = pm.sample(1000, tune=1000, return_inferencedata=True)
    
    def predict(self, X):
        """Predict survival times with uncertainty"""
        if self.trace is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Sample from posterior predictive distribution
        with self.model:
            ppc = pm.sample_posterior_predictive(self.trace, samples=1000)
        
        return ppc

def create_data_loaders():
    """Create train/test data loaders"""
    all_patients = labels_df.index.tolist()
    
    # Split into train (80%) and test (20%)
    print(f"Splitting data into train and test sets...")
    train_ids, test_ids = train_test_split(all_patients, test_size=0.2, random_state=seed_value)
    
    train_df = labels_df.loc[train_ids]
    test_df = labels_df.loc[test_ids]
    
    # Create datasets
    train_dataset = MultimodalMILBag(train_df, DATA_DIR, predict_type)
    test_dataset = MultimodalMILBag(test_df, DATA_DIR, predict_type)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return train_loader, test_loader

def train_attention_model(train_loader, val_loader):
    """Train the attention-based MIL model"""
    print("Training attention-based MIL model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BayesianAttentionMIL(
        input_dim=1024, 
        hidden_dim=args.hidden_dim, 
        predict_type=predict_type,
        dropout_rate=args.dropout_rate
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for tiles, *labels in train_loader:
            tiles = tiles.squeeze(0).to(device)
            labels = [label.to(device) for label in labels]
            
            optimizer.zero_grad()
            
            if predict_type == "recurrence_status":
                event = labels[0]
                risk, _ = model(tiles, training=True)
                loss = bce_loss(risk.unsqueeze(0), event.unsqueeze(0))
            else:
                time, event = labels
                risk, _ = model(tiles, training=True)
                loss = cox_loss(risk.unsqueeze(0), time.unsqueeze(0), event.unsqueeze(0))
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for tiles, *labels in val_loader:
                tiles = tiles.squeeze(0).to(device)
                labels = [label.to(device) for label in labels]
                
                if predict_type == "recurrence_status":
                    event = labels[0]
                    risk, _ = model(tiles, training=False)
                    loss = bce_loss(risk.unsqueeze(0), event.unsqueeze(0))
                else:
                    time, event = labels
                    risk, _ = model(tiles, training=False)
                    loss = cox_loss(risk.unsqueeze(0), time.unsqueeze(0), event.unsqueeze(0))
                
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("✓ Model saved.")
    
    return model

def extract_features_with_uncertainty(model, data_loader):
    """Extract features using attention model with uncertainty estimation"""
    print("Extracting features with uncertainty estimation...")
    
    device = next(model.parameters()).device
    features_list = []
    attention_maps_list = []
    uncertainties_list = []
    
    model.eval()
    with torch.no_grad():
        for tiles, *labels in data_loader:
            tiles = tiles.squeeze(0).to(device)
            
            if use_mc_dropout:
                # Monte Carlo prediction for uncertainty
                mc_predictions, mc_attention = model.mc_predict(tiles, n_samples=n_mc_samples)
                
                # Calculate uncertainty metrics
                mean_prediction = np.mean(mc_predictions)
                uncertainty = np.std(mc_predictions)
                
                # Use mean attention map
                mean_attention = np.mean(mc_attention, axis=0)
                
                features_list.append(mean_prediction)
                attention_maps_list.append(mean_attention)
                uncertainties_list.append(uncertainty)
            else:
                # Single prediction
                prediction, attention = model(tiles, training=False)
                features_list.append(prediction.item())
                attention_maps_list.append(attention.cpu().numpy())
                uncertainties_list.append(0.0)  # No uncertainty estimate
    
    return np.array(features_list), np.array(attention_maps_list), np.array(uncertainties_list)

def train_bayesian_survival_model(train_features, train_times, train_events):
    """Train Bayesian survival model on extracted features"""
    print("Training Bayesian survival model...")
    
    # Standardize features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features.reshape(-1, 1))
    
    # Fit Bayesian survival model
    bayesian_model = BayesianSurvivalModel(survival_model=survival_model)
    bayesian_model.fit(train_features_scaled, train_times, train_events)
    
    return bayesian_model, scaler

def evaluate_with_uncertainty(model, bayesian_model, scaler, test_loader, test_times, test_events):
    """Evaluate model with uncertainty quantification"""
    print("Evaluating with uncertainty quantification...")
    
    # Extract features with uncertainty
    test_features, test_attention, test_uncertainties = extract_features_with_uncertainty(model, test_loader)
    
    # Standardize features
    test_features_scaled = scaler.transform(test_features.reshape(-1, 1))
    
    # Get predictions from Bayesian model
    if predict_type == "recurrence_status":
        # For classification, use MC dropout uncertainty
        predictions = test_features
        probabilities = 1 / (1 + np.exp(-predictions))  # Sigmoid
        
        # Calculate accuracy
        predicted_labels = (probabilities > 0.5).astype(int)
        accuracy = np.mean(predicted_labels == test_events)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Mean Uncertainty: {np.mean(test_uncertainties):.4f}")
        
        return accuracy, predictions, probabilities, test_uncertainties, test_attention
    else:
        # For survival analysis, use Bayesian model predictions
        predictions = bayesian_model.predict(test_features_scaled)
        
        # Calculate concordance index
        c_index = concordance_index(test_times, predictions, test_events)
        
        print(f"Test Concordance Index: {c_index:.4f}")
        print(f"Mean Uncertainty: {np.mean(test_uncertainties):.4f}")
        
        return c_index, predictions, None, test_uncertainties, test_attention

def bootstrap_uncertainty_estimation(model, bayesian_model, scaler, test_loader, n_bootstraps=100):
    """Estimate uncertainty using bootstrap sampling"""
    print(f"Performing bootstrap uncertainty estimation with {n_bootstraps} resamples...")
    
    bootstrap_scores = []
    bootstrap_uncertainties = []
    
    for i in range(n_bootstraps):
        if i % 10 == 0:
            print(f"Bootstrap iteration {i+1}/{n_bootstraps}")
        
        # Bootstrap sample
        bootstrap_indices = np.random.choice(
            len(test_loader.dataset.patients), 
            size=len(test_loader.dataset.patients), 
            replace=True
        )
        
        bootstrap_patients = [test_loader.dataset.patients[i] for i in bootstrap_indices]
        bootstrap_subset = labels_df.loc[bootstrap_patients]
        bootstrap_dataset = MultimodalMILBag(bootstrap_subset, DATA_DIR, predict_type)
        bootstrap_loader = DataLoader(bootstrap_dataset, batch_size=1, shuffle=False)
        
        # Get bootstrap times and events
        bootstrap_times = bootstrap_subset['recurrence_time'].values
        bootstrap_events = bootstrap_subset['recurrence_status'].values
        
        # Evaluate on bootstrap sample
        if predict_type == "recurrence_status":
            bootstrap_accuracy, _, _, bootstrap_uncertainty, _ = evaluate_with_uncertainty(
                model, bayesian_model, scaler, bootstrap_loader, bootstrap_times, bootstrap_events
            )
            bootstrap_scores.append(bootstrap_accuracy)
            bootstrap_uncertainties.append(np.mean(bootstrap_uncertainty))
        else:
            bootstrap_c_index, _, _, bootstrap_uncertainty, _ = evaluate_with_uncertainty(
                model, bayesian_model, scaler, bootstrap_loader, bootstrap_times, bootstrap_events
            )
            bootstrap_scores.append(bootstrap_c_index)
            bootstrap_uncertainties.append(np.mean(bootstrap_uncertainty))
    
    # Calculate confidence intervals
    bootstrap_scores = np.array(bootstrap_scores)
    bootstrap_uncertainties = np.array(bootstrap_uncertainties)
    
    ci_lower = np.percentile(bootstrap_scores, 2.5)
    ci_upper = np.percentile(bootstrap_scores, 97.5)
    
    print(f"Bootstrap Results:")
    print(f"Mean Score: {np.mean(bootstrap_scores):.4f}")
    print(f"Score 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"Mean Uncertainty: {np.mean(bootstrap_uncertainties):.4f}")
    
    return bootstrap_scores, bootstrap_uncertainties

def plot_uncertainty_analysis(predictions, uncertainties, attention_maps, test_times, test_events):
    """Plot uncertainty analysis"""
    print("Plotting uncertainty analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Predictions vs Uncertainty
    axes[0, 0].scatter(predictions, uncertainties, alpha=0.6)
    axes[0, 0].set_xlabel('Predictions')
    axes[0, 0].set_ylabel('Uncertainty')
    axes[0, 0].set_title('Predictions vs Uncertainty')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Uncertainty distribution
    axes[0, 1].hist(uncertainties, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Uncertainty')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Uncertainty Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Attention map heatmap (mean across patients)
    mean_attention = np.mean(attention_maps, axis=0)
    im = axes[1, 0].imshow(mean_attention.T, cmap='viridis', aspect='auto')
    axes[1, 0].set_xlabel('Patients')
    axes[1, 0].set_ylabel('Tiles')
    axes[1, 0].set_title('Mean Attention Weights')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Plot 4: Survival curves with uncertainty (if survival analysis)
    if predict_type == "recurrence_time":
        # Sort by uncertainty
        sorted_indices = np.argsort(uncertainties)
        low_uncertainty = sorted_indices[:len(sorted_indices)//2]
        high_uncertainty = sorted_indices[len(sorted_indices)//2:]
        
        # Assuming KaplanMeierFitter is available, otherwise this will cause an error
        # from lifelines.plotting import KaplanMeierFitter
        # kmf_low = KaplanMeierFitter()
        # kmf_high = KaplanMeierFitter()
        
        # kmf_low.fit(test_times[low_uncertainty], test_events[low_uncertainty], label="Low Uncertainty")
        # kmf_high.fit(test_times[high_uncertainty], test_events[high_uncertainty], label="High Uncertainty")
        
        # kmf_low.plot_survival_function(ax=axes[1, 1])
        # kmf_high.plot_survival_function(ax=axes[1, 1])
        axes[1, 1].set_title('Survival Curves by Uncertainty')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # For classification, plot ROC curve with uncertainty
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(test_events, predictions)
        roc_auc = auc(fpr, tpr)
        
        axes[1, 1].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
        axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1, 1].set_xlabel('False Positive Rate')
        axes[1, 1].set_ylabel('True Positive Rate')
        axes[1, 1].set_title('ROC Curve')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'uncertainty_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training and evaluation function"""
    print(f"Starting Bayesian MIL model training...")
    print(f"Prediction type: {predict_type}")
    print(f"Survival model: {survival_model}")
    print(f"Mode: {mode}")
    print(f"MC Dropout: {use_mc_dropout}")
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders()
    
    # Split train into train/val for attention model training
    train_patients = train_loader.dataset.patients
    train_train_ids, val_ids = train_test_split(train_patients, test_size=0.2, random_state=seed_value)
    
    train_train_df = labels_df.loc[train_train_ids]
    val_df = labels_df.loc[val_ids]
    
    train_train_dataset = MultimodalMILBag(train_train_df, DATA_DIR, predict_type)
    val_dataset = MultimodalMILBag(val_df, DATA_DIR, predict_type)
    
    train_train_loader = DataLoader(train_train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    if mode == 'train':
        # Train attention model
        attention_model = train_attention_model(train_train_loader, val_loader)
        
        # Extract features from training data
        train_features, train_attention, train_uncertainties = extract_features_with_uncertainty(
            attention_model, train_loader
        )
        
        # Get training times and events
        train_times = labels_df.loc[train_loader.dataset.patients, 'recurrence_time'].values
        train_events = labels_df.loc[train_loader.dataset.patients, 'recurrence_status'].values
        
        # Train Bayesian survival model
        bayesian_model, scaler = train_bayesian_survival_model(train_features, train_times, train_events)
        
        # Save models
        model_data = {
            'attention_model_state': attention_model.state_dict(),
            'bayesian_model': bayesian_model,
            'scaler': scaler
        }
        torch.save(model_data, MODEL_SAVE_PATH)
        print(f"Models saved to {MODEL_SAVE_PATH}")
        
        # Evaluate on test set
        test_times = labels_df.loc[test_loader.dataset.patients, 'recurrence_time'].values
        test_events = labels_df.loc[test_loader.dataset.patients, 'recurrence_status'].values
        
        score, predictions, probabilities, uncertainties, attention_maps = evaluate_with_uncertainty(
            attention_model, bayesian_model, scaler, test_loader, test_times, test_events
        )
        
        if predict_type == "recurrence_status":
            print(f"Final Test Accuracy: {score:.4f}")
        else:
            print(f"Final Test Concordance Index: {score:.4f}")
        
        # Bootstrap uncertainty estimation
        if args.do_bootstrap:
            bootstrap_scores, bootstrap_uncertainties = bootstrap_uncertainty_estimation(
                attention_model, bayesian_model, scaler, test_loader, args.n_bootstraps
            )
        
        # Plot uncertainty analysis
        if args.plot_uncertainty:
            plot_uncertainty_analysis(predictions, uncertainties, attention_maps, test_times, test_events)
    
    elif mode == 'test':
        # Load trained models
        model_data = torch.load(MODEL_SAVE_PATH, map_location='cpu')
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        attention_model = BayesianAttentionMIL(
            input_dim=1024, 
            hidden_dim=args.hidden_dim, 
            predict_type=predict_type,
            dropout_rate=args.dropout_rate
        ).to(device)
        attention_model.load_state_dict(model_data['attention_model_state'])
        
        bayesian_model = model_data['bayesian_model']
        scaler = model_data['scaler']
        
        # Evaluate on test set
        test_times = labels_df.loc[test_loader.dataset.patients, 'recurrence_time'].values
        test_events = labels_df.loc[test_loader.dataset.patients, 'recurrence_status'].values
        
        score, predictions, probabilities, uncertainties, attention_maps = evaluate_with_uncertainty(
            attention_model, bayesian_model, scaler, test_loader, test_times, test_events
        )
        
        if predict_type == "recurrence_status":
            print(f"Test Accuracy: {score:.4f}")
        else:
            print(f"Test Concordance Index: {score:.4f}")
    
    else:
        raise ValueError("Mode must be either 'train' or 'test'")

if __name__ == "__main__":
    main()