import torch
from torch import optim
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
import torch.nn.functional as F
import mlflow
import torch.nn as nn


class SimpleCNN1D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SimpleCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) #Try adaptive pooling
        
        # Calculate the flatten size dynamically
        self.flatten_size = self._get_flatten_size(in_channels, 7500)
        
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def _get_flatten_size(self, in_channels, input_length):
        x = torch.zeros(1, in_channels, input_length)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x.numel()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.flatten_size)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EnhancedCNN1D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(EnhancedCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.2)
        
        self.flatten_size = self._get_flatten_size(in_channels, 7500)
        
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def _get_flatten_size(self, in_channels, input_length):
        x = torch.zeros(1, in_channels, input_length)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x.numel()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.flatten_size)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ExampleNetwork(LightningModule):
    def __init__(self, model, data_loader, val_data_loader):
        super(ExampleNetwork, self).__init__()
        self.mdl: torch.nn.Module = model
        self.data_loader: DataLoader = data_loader
        self.val_data_loader: DataLoader = val_data_loader

        self.lr = 0.001
        self.batch_size = data_loader.batch_size
        mlflow.set_experiment("training_pipeline")
        mlflow.start_run(run_name = f"Efficient Net Classifier",nested=True)

    def forward(self, x: torch.Tensor):
        return self.mdl(x.float())

    def predict(self, x: torch.Tensor):
        with torch.no_grad():
            out = self.forward(x.float())
        return out

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        return self.data_loader

    def val_dataloader(self):
        return self.val_data_loader

    def training_step(self, batch: torch.Tensor, batch_nb: int):
        x, y = batch
        y = torch.squeeze(y.to(torch.int64))
        loss = F.cross_entropy(self(x.float()), y)
        mlflow.log_metric("loss", loss, step=self.global_step)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_nb: int):
        x, y = batch
        y = torch.squeeze(y.to(torch.int64))
        loss = F.cross_entropy(self(x.float()), y)
        mlflow.log_metric("val_loss", loss, step=batch_nb*(self.current_epoch+1) )
        return loss

    def on_fit_end(self):
        mlflow.end_run()