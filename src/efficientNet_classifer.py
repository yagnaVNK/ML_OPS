import torch
from torch import optim
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
import torch.nn.functional as F


class ExampleNetwork(LightningModule):
    def __init__(self, model, data_loader, val_data_loader):
        super(ExampleNetwork, self).__init__()
        self.mdl: torch.nn.Module = model
        self.data_loader: DataLoader = data_loader
        self.val_data_loader: DataLoader = val_data_loader

        self.lr = 0.001
        self.batch_size = data_loader.batch_size

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
        self.log("loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_nb: int):
        x, y = batch
        y = torch.squeeze(y.to(torch.int64))
        loss = F.cross_entropy(self(x.float()), y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
