import logging
from zenml import step
from torch.utils.data import DataLoader
from src.efficientNet_classifer import *
import torch
from torchsig.models.iq_models.efficientnet.efficientnet import efficientnet_b4, create_effnet
from src.utils import *
from zenml.client import Client
import lightning.pytorch as pl
import timm
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import MLFlowExperimentTrackerSettings

mlflow_settings = MLFlowExperimentTrackerSettings(
    nested=True,
    tags={"Cos Loss": "0.7"}
)

experiment_tracker = Client().active_stack.experiment_tracker



@step(enable_cache=True,experiment_tracker = experiment_tracker.name,
      settings={
        "experiment_tracker.mlflow": mlflow_settings
    })
def train_classifier(dl_train: DataLoader,
                dl_val: DataLoader,
                epochs: int,
                train_bool: bool,
                eff_net_PATH: str,
                classes: list,
                in_channels: int) -> ExampleNetwork:
    
    model = create_effnet(
        timm.create_model(
            "efficientnet_b4",
            num_classes=len(classes),
            in_chans=in_channels,
        )
    )
    
    example_model = ExampleNetwork(model, dl_train, dl_val)
    example_model = example_model.float().to(device)    
    trainer = Trainer(
        max_epochs=epochs, 
        devices=1, 
        accelerator="gpu"
    )
    if train_bool:
        trainer.fit(example_model,dl_train,dl_val)
        torch.save(example_model.state_dict(),eff_net_PATH)
        print("trained the model")
        return example_model
    else:
        example_model.load_state_dict(torch.load(eff_net_PATH))
        print("loaded from checkpoint")
        return example_model
