# Multimodal MIL model for GBM recurrence prediction using synthetic tile embeddings
# num_patients = 17, num_tiles = 2000, embedding_dim = 512
# total number of tiles = 17 * 2000 = 34000

# Steps involving MIL:
# 1. Learn which tiles are most important for predicting the bag-level label
# 2. Aggregate the tile embeddings into a single, meaningful patient-level embedding
# 3. Use the patient-level embedding to predict the bag-level label

# To do:
# 1. Add cross-modal attention to account for interaction between modalities instead of just concatenating the embeddings
# 2. Tile level interpretability: extract the most important tiles for each patient by visualizing the attention weights
# 3. Add surival model for recurrence time prediction (to account for right-censoring)

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pdb import set_trace

# need to use survival model and not MSE since we are using time to recurrence as the target that is not continuous, not normally distributed, and can be right-censored due to recurrence status being 0 for some patients
predict_type = "recurrence_status"  # "recurrence_status" or "recurrence_time"
use_cross_attention = False        # Set to False to disable cross-modal attention (can be problematic for small dataset size)
num_epochs = 100
mode = 'train' # 'train' or 'test'

DATA_DIR = './synthetic_embeddings' # directory containing the synthetic embeddings
MODEL_SAVE_PATH = './models/mil_model.pt' # directory to save the model

# Load labels
labels_df = pd.read_csv(os.path.join(DATA_DIR, 'labels.csv'))
labels_df.set_index('patient_id', inplace=True)

# Dataset class
# returns tiles and associated labels for each patient for each modality
class MultimodalMILBag(Dataset):
    def __init__(self, labels_df, data_dir, predict_type="recurrence_status"):
        super().__init__()
        self.labels_df = labels_df
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

        # Prepare dictionary of modality embeddings
        tiles_dict = {
            'histology': hist_tensor,
            'spatial_rnaseq': st_tensor
        }

        # Load label
        if self.predict_type == "recurrence_status":
            label = self.labels_df.loc[patient_id, 'recurrence_status']
        elif self.predict_type == "recurrence_time":
            label = self.labels_df.loc[patient_id, 'recurrence_time']
        else:
            raise ValueError(f"Unsupported predict_type: {self.predict_type}")

        label_tensor = torch.tensor(label, dtype=torch.float32)

        return tiles_dict, label_tensor


# model definition
# Attention-based MIL model to learn which tiles (instances) in a bag (patient sample) are most important 
# for predicting the bag-level label, and then to aggregate the tile embeddings into a single, meaningful patient-level embedding.
# Note: in standard pooling (e.g., mean, max), all tiles are treated equally, regardless of their importance
class AttentionMIL(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128):
        super().__init__()
        self.attention_V = nn.Linear(input_dim, hidden_dim) # converts each tile embedding to hidden_dim-dimensional vector
        self.attention_w = nn.Linear(hidden_dim, 1) # converts hidden_dim-dimensional vector to a single scalar (attention weight)

    def forward(self, x):  # x: [num_tiles, input_dim]
        A = torch.tanh(self.attention_V(x))         # [num_tiles, hidden_dim]
        A = self.attention_w(A)                     # [num_tiles, 1]
        A = torch.softmax(A, dim=0)                 # attention weights
        z = torch.sum(A * x, dim=0)                 # weighted sum: [input_dim]
        return z, A # aggregated embedding of shape [input_dim], attention weights for each tile of shape [num_tiles, 1]

class MultiModalMILModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128):
        super().__init__()
        self.mil_hist = AttentionMIL(input_dim, hidden_dim)
        self.mil_st = AttentionMIL(input_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, tiles_dict):
        h_embedding, _ = self.mil_hist(tiles_dict['histology'])       # [embedding_dim]
        st_embedding, _ = self.mil_st(tiles_dict['spatial_rnaseq'])   # [embedding_dim]
        fused = torch.cat([h_embedding, st_embedding], dim=0)         # [2*embedding_dim]
        out = self.classifier(fused)                                  # [1]
        return out.squeeze()


all_patients = labels_df.index.tolist()

# Split into train (60%), val (20%), test (20%)
print(f"Splitting data into train, val, test sets...")
train_ids, test_ids = train_test_split(all_patients, test_size=0.4, random_state=42)
val_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=42)

train_df = labels_df.loc[train_ids]
val_df = labels_df.loc[val_ids]
test_df = labels_df.loc[test_ids]

# dataset and dataloader
print(f"Creating datasets...")
# Create datasets
train_dataset = MultimodalMILBag(train_df, DATA_DIR, predict_type)
val_dataset = MultimodalMILBag(val_df, DATA_DIR, predict_type)
test_dataset = MultimodalMILBag(test_df, DATA_DIR, predict_type)

print(f"Creating dataloaders...")
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True) # batch_size=1 since MIL is sample-level
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) 

print(f"Creating model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiModalMILModel().to(device)

# loss function
if predict_type == "recurrence_status": # binary classification
    criterion = nn.BCELoss()
elif predict_type == "recurrence_time":
    criterion = nn.BCELoss() # change this to survival loss

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# training loop
if mode == 'train':
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}...")
        model.train()
        train_loss = 0.0

        for tiles_dict, label in train_loader:
            tiles_dict = {k: v.squeeze(0).to(device) for k, v in tiles_dict.items()}
            label = label.to(device)

            output = model(tiles_dict)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for tiles_dict, label in val_loader:
                tiles_dict = {k: v.squeeze(0).to(device) for k, v in tiles_dict.items()}
                label = label.to(device)
                output = model(tiles_dict)
                loss = criterion(output, label)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("✓ Model saved.")

# evaluation on test set
elif mode == 'test':
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for tiles_dict, label in test_loader:
            tiles_dict = {k: v.squeeze(0).to(device) for k, v in tiles_dict.items()}
            label = label.to(device)

            output = model(tiles_dict)
            predicted = (output > 0.5).float()
            correct += (predicted == label).sum().item()
            total += 1

    print(f"Test Accuracy: {correct / total:.2f}")

else:
    raise ValueError("Mode must be either 'train' or 'test'")