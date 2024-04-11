import logging
import torchsig.transforms
from zenml import step
import torchsig
from torch.utils.data import DataLoader
from torchsig.datasets.modulations import ModulationsDataset
from torchsig.transforms.transforms import (
    RandomPhaseShift,
    Normalize,
    ComplexTo2D,
    Compose,
)
import torch
import torchsig.transforms as ST

@step
def getDataLoader(classes: list,
                   iq_samples: int, 
                   samples_per_class: int,
                   Train: bool,
                   batch_size: int) -> DataLoader:
    '''
    Returns data loaders and datasets for train, test and validation

    Args: 
        classes -> list of modulations to generate from the torchsig dataset. 
    '''
    data_transform = ST.Compose([
        ST.ComplexTo2D(),
    ])
    if Train:
        ds_train = ModulationsDataset(
            classes = classes,
            use_class_idx = True,
            level=0,
            num_iq_samples=iq_samples,
            num_samples=int(len(classes)*samples_per_class),
            include_snr=False,
            transform = data_transform
        )

        dl_train = DataLoader(ds_train,batch_size=batch_size,shuffle=True)

        return dl_train
    else:
        ds_test = ModulationsDataset(
            classes = classes,
            use_class_idx = True,
            level=0,
            num_iq_samples=iq_samples,
            num_samples=int(len(classes)*samples_per_class),
            include_snr=False,
            transform = data_transform
        )

        dl_test = DataLoader(ds_test,batch_size=batch_size,shuffle=False)

        return dl_test
