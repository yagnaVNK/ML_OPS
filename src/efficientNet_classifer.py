import torch
from torch import optim
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
import torch.nn.functional as F
import mlflow


class ExampleNetwork(LightningModule):
    def __init__(self, model, data_loader, val_data_loader,experimentName):
        super(ExampleNetwork, self).__init__()
        self.mdl: torch.nn.Module = model
        self.data_loader: DataLoader = data_loader
        self.val_data_loader: DataLoader = val_data_loader

        self.lr = 0.001
        self.batch_size = data_loader.batch_size
        mlflow.set_experiment(experimentName)
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