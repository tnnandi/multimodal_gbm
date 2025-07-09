# This script generates synthetic tile-level embedding data for a multimodal MIL model.
# Each patient has 2000 tiles per modality, and each tile has a 512-dimensional embedding.

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

# Configuration
np.random.seed(42)  
num_patients = 17
num_tiles = 2000
embedding_dim = 512
modalities = ['histology', 'spatial_rnaseq']
output_dir = 'synthetic_embeddings'
os.makedirs(output_dir, exist_ok=True)

# Labels: Recurrence status and time
patient_labels = {}

# Glioblastoma realistic recurrence: 60-80% recur, median ~8 months
recurrence_prob = 0.7  # 70% patients recur
min_time = 1
max_time = 36  # months

for patient_id in range(1, num_patients + 1):
    patient_key = f"patient_{patient_id:02d}"

    # Assign recurrence
    has_recurred = int(np.random.rand() < recurrence_prob)

    # If recurred, assign a realistic recurrence time (skewed toward early recurrence)
    if has_recurred:
        # Use exponential distribution scaled to a median around 8 months
        recurrence_time = int(np.clip(np.random.exponential(scale=8), min_time, max_time))
    else:
        recurrence_time = np.nan  # no recurrence, censored

    # Save label
    patient_labels[patient_key] = {
        'recurrence_status': has_recurred,
        'recurrence_time': recurrence_time
    }

    # Save embeddings
    for modality in modalities:
        embeddings = np.random.randn(num_tiles, embedding_dim).astype(np.float32)
        df = pd.DataFrame(embeddings, columns=[f"dim_{i+1}" for i in range(embedding_dim)])
        df.insert(0, 'tile_id', [f"tile_{i+1}" for i in range(num_tiles)])
        file_path = os.path.join(output_dir, f"{patient_key}_{modality}.csv")
        df.to_csv(file_path, index=False)

# Save labels to CSV
labels_df = pd.DataFrame.from_dict(patient_labels, orient='index')
labels_df.index.name = 'patient_id'
labels_df.to_csv(os.path.join(output_dir, 'labels.csv'))

print(f"Synthetic data for {num_patients} patients created in '{output_dir}/'")

# Kaplan-Meier plot
kmf = KaplanMeierFitter()
observed = labels_df['recurrence_status'] == 1
durations = labels_df['recurrence_time'].fillna(max_time) # check if this is correct

plt.figure(figsize=(8, 5))
kmf.fit(durations, event_observed=observed, label='Recurrence-free survival')
kmf.plot()
plt.title('Synthetic Glioblastoma Recurrence KM Curve')
plt.xlabel('Time (months)')
plt.ylabel('Survival probability')
plt.grid(True)
plt.tight_layout()
plt.savefig('synthetic_km_curve.png')
# plt.show()

