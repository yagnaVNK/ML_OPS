import logging
from zenml import step
from torch.utils.data import DataLoader

@step
def train_model(dl_train: DataLoader,
                dl_val: DataLoader,
                epochs: int) -> None:
    """
    
    """
    pass