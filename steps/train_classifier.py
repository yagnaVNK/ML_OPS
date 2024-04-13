import logging
from zenml import step
from torch.utils.data import DataLoader
from src.efficientNet_classifer import *
import torch
from torchsig.models.iq_models.efficientnet.efficientnet import efficientnet_b4
from src.utils import *

@step
def train_classifier(dl_train: DataLoader,
                dl_val: DataLoader,
                epochs: int,
                train_bool: bool,
                eff_net_PATH: str) -> ExampleNetwork:
    
    model = efficientnet_b4(
        pretrained=False
    )
    example_model = ExampleNetwork(model, dl_train, dl_val)
    example_model = example_model.float().to(device)    
    trainer = Trainer(
        max_epochs=epochs, 
        devices=1, accelerator="gpu"
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
