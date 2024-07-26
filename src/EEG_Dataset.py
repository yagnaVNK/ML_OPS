import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        self.data = []
        self.labels = []
        self.load_data(data_dir, split)

    def load_data(self, data_dir, split):
        files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.pkl')]
        
        # Split files into train and test sets
        train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
        # Further split train files into train and validation sets
        train_files, val_files = train_test_split(train_files, test_size=0.2, random_state=42)

        if split == "train":
            selected_files = train_files
        elif split == "val":
            selected_files = val_files
        else:  # split == "test"
            selected_files = test_files

        for file in selected_files:
            with open(file, 'rb') as f:
                data = pickle.load(f)
                data = data.transpose(1, 2, 0).reshape(32, 7500, 28)
                for idx in range(28):
                    self.data.append(data[:, :, idx])
                    self.labels.append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample, label