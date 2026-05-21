import numpy as np 
import torch
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.model_selection import train_test_split
import json

LANDMARK_DIR = Path("data/landmarks")

def load_all_classes():
    X_list = []
    Y_list = []
    label_map = {}
    
    npy_files = sorted(LANDMARK_DIR.glob("*.npy"))
    for idx, npy_paths in enumerate(npy_files):
        letter = npy_paths.stem.upper()
        data = np.load(npy_paths)   # shape (n_samples, 126)
        
        label_map[idx] = letter
        X_list.append(data)
        Y_list.extend([idx] * len(data))
        
        print(f"  Loaded {letter}: {len(data)} samples (label={idx})")

    X = np.vstack(X_list)          # shape: (total_samples, 126)
    y = np.array(Y_list)           # shape: (total_samples,)

    print(f"\nTotal: {len(X)} samples across {len(label_map)} classes")
    return X, y, label_map

def split_data(X, y, val_size=0.15, test_size=0.15, random_state=42):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    adjusted_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=adjusted_val,
        stratify=y_train_val,
        random_state=random_state
    )

    print(f"Split sizes → train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test

class BisindoDataset(Dataset):
    """
    PyTorch Dataset wrapping numpy arrays.
    Converts to float32 tensors on access — DataLoader will batch these automatically.
    """
    def __init__(self, X, y):
        # float32 is required by PyTorch linear layers; numpy defaults to float64
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)    # long (int64) required for CrossEntropyLoss

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]   # returns one (features, label) pair

if __name__ == "__main__":
    # Quick sanity check — run this file directly to verify loading and splits
    X, y, label_map = load_all_classes()

    # Save label_map as JSON — this goes into bisindo-space/ later for inference
    with open("label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"\nSaved label_map.json: {label_map}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Verify tensor shapes before plugging into model
    dataset = BisindoDataset(X_train, y_train)
    sample_X, sample_y = dataset[0]
    print(f"\nSample tensor shapes — X: {sample_X.shape}, y: {sample_y.shape}")
    print(f"X dtype: {sample_X.dtype}, y dtype: {sample_y.dtype}")