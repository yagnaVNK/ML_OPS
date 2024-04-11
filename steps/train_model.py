import logging
from zenml import step
from torch.utils.data import DataLoader
from src.HAE import *
import torch

@step
def train_model(dl_train: DataLoader,
                dl_val: DataLoader,
                epochs: int) -> None:
    """
    
    """
    enc_hidden_sizes = [16, 16, 32, 64, 128]
    dec_hidden_sizes = [16, 64, 256, 512, 1024]
    layers = 2
    KL_coeff = 0.1
    batch_norm = 1
    for i in range(layers): 
        print(f'training Layer {i}')
        print('==============================================')
        if i == 0:
            hae = HAE.init_bottom(
                input_feat_dim=2,
                enc_hidden_dim=enc_hidden_sizes[i],
                dec_hidden_dim=dec_hidden_sizes[i],
                num_res_blocks=0,
                KL_coeff = KL_coeff,
                batch_norm = batch_norm,
                layer = i,
                cos_reset = 0,
                compress = 2
            )
            
        else:
            hae = HAE.init_higher(
                hae_prev,
                enc_hidden_dim=enc_hidden_sizes[i],
                dec_hidden_dim=dec_hidden_sizes[i],
                num_res_blocks=2,
                KL_coeff = KL_coeff,
                batch_norm = batch_norm,
                layer = i,
                cos_reset = 0,
                compress = 2
            )
        trainer = pl.Trainer(max_epochs=epochs, 
             logger=None,  
             devices=1,
             accelerator = 'gpu',
             num_sanity_val_steps=0,
        )
        trainer.fit(hae.float(), dl_train, dl_val)
        hae_prev = hae.eval()
    return None