import logging
import torchsig.transforms
from zenml import step
import torchsig
from torch.utils.data import DataLoader
from torchsig.datasets.modulations import ModulationsDataset
from typing_extensions import Annotated
from typing import Tuple
import torch
import torchsig.transforms as ST

@step(enable_cache=True)
def getDataLoader(classes: list,
                   iq_samples: int, 
                   samples_per_class: int,
                   batch_size: int) -> Tuple[
                       Annotated[DataLoader,"train_dl"],
                       Annotated[ModulationsDataset,"train_ds"],
                       Annotated[DataLoader,"test_dl"],
                       Annotated[ModulationsDataset,"test_ds"],
                       Annotated[DataLoader,"val_dl"],
                       Annotated[ModulationsDataset,"val_ds"],
                       ]:
    '''
    Returns data loaders and datasets for train, test and validation

    Args: 
        classes -> list of modulations to generate from the torchsig dataset. 
    '''
    data_transform = ST.Compose([
        ST.ComplexTo2D(),
    ])
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
    ds_val = ModulationsDataset(
        classes = classes,
        use_class_idx = True,
        level=0,
        num_iq_samples=iq_samples,
        num_samples=int(len(classes)*samples_per_class)/80,
        include_snr=False,
        transform = data_transform
    )

    dl_val = DataLoader(ds_val,batch_size=batch_size,shuffle=False)

    ds_test = ModulationsDataset(
        classes = classes,
        use_class_idx = True,
        level=0,
        num_iq_samples=iq_samples,
        num_samples=int(len(classes)*samples_per_class)/80,
        include_snr=False,
        transform = data_transform
    )

    dl_test = DataLoader(ds_test,batch_size=batch_size,shuffle=False)



    return dl_train,ds_train,dl_test,ds_test,dl_val,ds_val

