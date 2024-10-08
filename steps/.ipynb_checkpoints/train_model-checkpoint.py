from zenml import step
from torch.utils.data import DataLoader
from src.HAE import *
from src.HQA import *
from lightning.pytorch.loggers import MLFlowLogger
import mlflow
from zenml.client import Client
import lightning.pytorch as pl
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import MLFlowExperimentTrackerSettings

mlflow_settings = MLFlowExperimentTrackerSettings(
    nested=True,
    tags={"key": "value"}
)

experiment_tracker = Client().active_stack.experiment_tracker


@step(enable_cache=False,experiment_tracker = experiment_tracker.name,
      settings={
        "experiment_tracker.mlflow": mlflow_settings
    })
def train_HAE(dl_train: DataLoader,
              dl_val: DataLoader,
              epochs: int,
              layers: int,
              enc_hidden_sizes: list,
              dec_hidden_sizes: list,
              input_feat_dim: int,
              codeword_dim: int,
              batch_norm: int,
              Cos_coeff:float,
              num_res_blocks: int,
              cos_reset: int,
              compress: int,
              train_flag: bool,
              hae_lr:float) -> HAE:
    """

    """
    enc_hidden_sizes = enc_hidden_sizes
    dec_hidden_sizes = dec_hidden_sizes
    layers = layers
    batch_norm = batch_norm
    
    for i in range(layers): 
        print(f'training Layer {i}')
        print('===============================================')
        if i == 0:
            hae = HAE.init_bottom(
                input_feat_dim=input_feat_dim,
                enc_hidden_dim=enc_hidden_sizes[i],
                dec_hidden_dim=dec_hidden_sizes[i],
                num_res_blocks=num_res_blocks,
                batch_norm = batch_norm,
                layer = i,
                cos_reset = cos_reset,
                compress = compress,
                codebook_dim = codeword_dim,
                Cos_coeff = Cos_coeff,
                train_flag = train_flag,
                lr=hae_lr
            )
            
        else:
            hae = HAE.init_higher(
                hae_prev,
                enc_hidden_dim=enc_hidden_sizes[i],
                dec_hidden_dim=dec_hidden_sizes[i],
                num_res_blocks=num_res_blocks ,
                batch_norm = batch_norm,
                layer = i,
                cos_reset = cos_reset,
                compress = compress,
                codebook_dim = codeword_dim,
                Cos_coeff = Cos_coeff,
                train_flag = train_flag,
                lr=hae_lr
            )
        
        trainer = pl.Trainer(max_epochs=epochs,logger=None, devices=[0], accelerator = 'gpu',num_sanity_val_steps=0)
        
        trainer.fit(hae.float(), dl_train, dl_val)
        hae_prev = hae.eval()
    return hae


@step(enable_cache=False,experiment_tracker = experiment_tracker.name,
      settings={
        "experiment_tracker.mlflow": mlflow_settings
    })
def train_HQA(dl_train: DataLoader,
              dl_val: DataLoader,
              epochs: int,
              layers: int,
              enc_hidden_sizes: list,
              dec_hidden_sizes: list,
              input_feat_dim: int,
              codeword_dim: int,
              codebook_slots:int,
              codebook_init: str,
              output_dir: str,
              codeword_reset: int,
              KL_coeff:float,
              CL_coeff:float,
              Cos_coeff:float,
              batch_norm: int,
              num_res_blocks: int,
              cos_reset: int,
              compress: int,
              model:HAE,
              hqa_lr:float) -> HQA:
    """
    
    """
    model = model.to('cuda')
    for i in range(layers): 
        print(f'training Layer {i}')
        print("==============================================")
        if i == 0:
            hqa = HQA.init_bottom(
                input_feat_dim=input_feat_dim,
                enc_hidden_dim=enc_hidden_sizes[i],
                dec_hidden_dim=dec_hidden_sizes[i],
                num_res_blocks=num_res_blocks,
                KL_coeff = KL_coeff,
                CL_coeff = CL_coeff,
                Cos_coeff = Cos_coeff,
                batch_norm = batch_norm,
                codebook_init = codebook_init,
                reset_choice = codeword_reset,
                output_dir = output_dir,
                codebook_slots = codebook_slots,
                codebook_dim = codeword_dim,
                layer = i,
                lr = hqa_lr,
                cos_reset = cos_reset,
                compress = compress,
                train_dataloader = dl_train
            )
            
        else:
            hqa = HQA.init_higher(
                hqa_prev,
                enc_hidden_dim=enc_hidden_sizes[i],
                dec_hidden_dim=dec_hidden_sizes[i],
                num_res_blocks=num_res_blocks  ,
                KL_coeff = KL_coeff,
                CL_coeff = CL_coeff,
                Cos_coeff = Cos_coeff,
                batch_norm = batch_norm,
                codebook_init = codebook_init,
                reset_choice = codeword_reset,
                output_dir = output_dir,
                codebook_slots = codebook_slots,
                codebook_dim = codeword_dim,
                layer = i,
                lr = hqa_lr,
                cos_reset = cos_reset,
                compress = compress,
                train_dataloader = dl_train
            )
        hqa.encoder = model[i].encoder.eval()
        hqa.decoder = model[i].decoder.eval()
        print("loaded the encoder and decoder pretrained models")

        trainer = pl.Trainer(max_epochs=epochs,logger=None, devices=[0], accelerator = 'gpu',num_sanity_val_steps=0)
        trainer.fit(hqa.float(), dl_train, dl_val)
        hqa_prev = hqa.eval()

    return hqa