import logging
from zenml import step
from torch.utils.data import DataLoader

@step
def eval_model(dl_test:DataLoader) -> None:
    """
    
    """
    pass