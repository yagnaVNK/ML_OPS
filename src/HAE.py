import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.distributions import RelaxedOneHotCategorical, Normal, Categorical
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torchvision import transforms
from torchvision.datasets import MNIST
import torchsig.transforms as ST
from torchsig.datasets.modulations import ModulationsDataset
import lightning.pytorch as pl
import argparse
from lightning.pytorch.loggers import TensorBoardLogger
from torchsummary import summary
from torch.nn import GELU
from sklearn.manifold import TSNE
import mlflow


def mish(x):
    return x * torch.tanh(F.softplus(x))

class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return mish(x)

class FlatCA(_LRScheduler):
    def __init__(self, optimizer, steps, eta_min=0, last_epoch=-1):
        self.steps = steps
        self.eta_min = eta_min
        super(FlatCA, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lr_list = []
        T_max = self.steps / 3
        for base_lr in self.base_lrs:
            # flat if first 2/3
            if 0 <= self._step_count < 2 * T_max:
                lr_list.append(base_lr)
            # annealed if last 1/3
            else:
                lr_list.append(
                    self.eta_min
                    + (base_lr - self.eta_min)
                    * (1 + math.cos(math.pi * (self._step_count - 2 * T_max) / T_max))
                    / 2
                )
            return lr_list


class Encoder(nn.Module):
    """ Downsamples by a fac of 2 """

    def __init__(self, in_feat_dim, codebook_dim, hidden_dim=128, num_res_blocks=0, batch_norm=1):
        super().__init__()
        blocks = [
            nn.Conv1d(in_feat_dim, hidden_dim // 2, kernel_size=3, stride=2, padding=1),
            Mish(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            Mish(),
        ]

        for _ in range(num_res_blocks):
            blocks.append(ResBlock(hidden_dim, hidden_dim // 2))
            if batch_norm == 2:
                blocks.append(nn.BatchNorm1d(hidden_dim))

        blocks.append(nn.Conv1d(hidden_dim, codebook_dim, kernel_size=1))
        if(batch_norm):
            blocks.append(nn.BatchNorm1d(codebook_dim))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = x.float()
        return self.blocks(x)

class Encoder2(nn.Module):
    """ Downsamples by a fac of 4 """
    def __init__(self, in_feat_dim, codebook_dim, hidden_dim=128, num_res_blocks=0,batch_norm=1):
        super().__init__()
        blocks = [
            nn.Conv1d(in_feat_dim, hidden_dim // 2, kernel_size=7, stride=4, padding=2),
            Mish(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            Mish(),
        ]

        for _ in range(num_res_blocks):
            blocks.append(ResBlock(hidden_dim, hidden_dim // 2))
            if batch_norm==2:
                blocks.append(nn.BatchNorm1d(hidden_dim))
        blocks.append(nn.Conv1d(hidden_dim, codebook_dim, kernel_size=1))

        if(batch_norm):
            blocks.append(nn.BatchNorm1d(codebook_dim))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = x.float()
        return self.blocks(x)



class Decoder(nn.Module):
    def __init__(
        self, in_feat_dim, out_feat_dim, hidden_dim=128, num_res_blocks=0, very_bottom=False,
    ):
        super().__init__()
        self.very_bottom = very_bottom
        self.out_feat_dim = out_feat_dim # num channels on bottom layer
        blocks = [nn.Conv1d(in_feat_dim, hidden_dim, kernel_size=3, padding=1), Mish()]
        for _ in range(num_res_blocks):
            blocks.append(ResBlock(hidden_dim, hidden_dim // 2))

        blocks.extend([
                Upsample(),
                nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
                Mish(),
                nn.Conv1d(hidden_dim // 2, out_feat_dim, kernel_size=3, padding=1),
        ])
 
        if very_bottom is True:
            blocks.append(nn.Tanh())       
        self.blocks = nn.Sequential(*blocks)
        
    def forward(self, x):
        x = x.float()
        return self.blocks(x)

class Decoder2(nn.Module):
    """ Upsamples by a fac of 2 """
    def __init__(
        self, in_feat_dim, out_feat_dim, hidden_dim=128, num_res_blocks=0, very_bottom=False,
    ):
        super().__init__()
        self.very_bottom = very_bottom
        self.out_feat_dim = out_feat_dim # num channels on bottom layer
        
        blocks = [nn.Conv1d(in_feat_dim, hidden_dim, kernel_size=3, padding=1), Mish()]

        for _ in range(num_res_blocks):
            blocks.append(ResBlock(hidden_dim, hidden_dim // 2))

        blocks.extend([
                Upsample(),
                nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
                Mish(),
                nn.Conv1d(hidden_dim // 2, out_feat_dim, kernel_size=3, padding=1),
                Upsample(),
        ])

        if very_bottom is True:
            blocks.append(nn.Tanh()) 
            
        self.blocks = nn.Sequential(*blocks)
        
    def forward(self, x):
        x = x.float()
        return self.blocks(x)

class Upsample(nn.Module):

    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor)

class ResBlock(nn.Module):

    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channel, channel, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv1d(channel, in_channel, kernel_size=3, padding=1)

    def forward(self, inp):
        x = self.conv_1(inp)
        x = mish(x)
        x = self.conv_2(x)
        x = x + inp
        return mish(x)


class GlobalNormalization1(torch.nn.Module):
    def __init__(self, feature_dim, scale=False):
        super().__init__()

        self.feature_dim = feature_dim
        self.register_buffer("running_ave", torch.zeros(1, self.feature_dim, 1))
        self.register_buffer("total_frames_seen", torch.Tensor([0]))
        self.scale = scale
        if self.scale:
            self.register_buffer("running_sq_diff", torch.zeros(1, self.feature_dim, 1))



    def forward(self, inputs):
        if self.training:
            # Update running estimates of statistics
            frames_in_input = inputs.shape[0] * inputs.shape[2]
            updated_running_ave = (
                self.running_ave * self.total_frames_seen + inputs.sum(dim=(0, 2), keepdim=True)
            ) / (self.total_frames_seen + frames_in_input)
            if self.scale:
                # Update the sum of the squared differences between inputs and mean
                self.running_sq_diff = self.running_sq_diff + (
                    (inputs - self.running_ave) * (inputs - updated_running_ave)
                ).sum(dim=(0, 2), keepdim=True)
            self.running_ave = updated_running_ave
            self.total_frames_seen = self.total_frames_seen + frames_in_input
        else:
            return inputs


        if self.scale:
            std = torch.sqrt(self.running_sq_diff / self.total_frames_seen)
            inputs = (inputs - self.running_ave) / std
        else:
            inputs = inputs - self.running_ave
        return inputs



    def unnorm(self, inputs):
        if self.scale:
            std = torch.sqrt(self.running_sq_diff / self.total_frames_seen)
            inputs = inputs*std + self.running_ave
        else:
            inputs = inputs + self.running_ave

        return inputs

  

class HAE(pl.LightningModule):

    VISUALIZATION_DIR = 'vis'
    SUBDIRS=[VISUALIZATION_DIR]

    def __init__(
        self,
        input_feat_dim,
        prev_model=None,
        codebook_dim=256,
        enc_hidden_dim=16,
        dec_hidden_dim=32,
        num_res_blocks=0,
        lr=4e-4,
        decay=True,
        clip_grads=False,
        layer = 0 ,
        Cos_coeff = 0.7,
        batch_norm = 1,
        cos_reset = 1,
        compress = 2,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['prev_model'])
        self.prev_model = prev_model
        
        self.batch_norm = batch_norm
        if compress ==2:
            self.encoder = Encoder(input_feat_dim, codebook_dim, enc_hidden_dim, num_res_blocks=num_res_blocks,batch_norm=self.batch_norm)
            self.decoder = Decoder(
                codebook_dim,
                input_feat_dim,
                dec_hidden_dim,
                very_bottom=prev_model is None,
                num_res_blocks=num_res_blocks
            )
        else:
            self.encoder = Encoder2(input_feat_dim, codebook_dim, enc_hidden_dim, num_res_blocks=num_res_blocks,batch_norm=self.batch_norm)
            self.decoder = Decoder2(
                codebook_dim,
                input_feat_dim,
                dec_hidden_dim,
                very_bottom=prev_model is None,
                num_res_blocks=num_res_blocks
            )
        self.normalize = GlobalNormalization1(codebook_dim, scale=True)
        self.out_feat_dim = input_feat_dim
        self.codebook_dim = codebook_dim
        self.lr = lr
        self.decay = decay
        self.clip_grads = clip_grads
        self.layer = layer
        self.Cos_coeff = torch.tensor(Cos_coeff)
        self.cos_reset = cos_reset
        self.train_outputs = {}
        self.val_outputs = {}
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.input_feat_dim = input_feat_dim
        self.compress = compress
        self.num_res_blocks = num_res_blocks
    
        # Tells pytorch lightinig to use our custom training loop
        self.automatic_optimization = False
        mlflow.set_experiment("training_pipeline")
        mlflow.start_run(run_name = f"HAE Layer {self.layer}",nested=True)
        # Log hyperparameters
        
                

    def forward(self, x, soft = True):
        if x.dtype != torch.float32:
            x = x.float()
        z_e_lower = self.encode_lower(x)
        z_e = self.encoder(z_e_lower)
        z_e_lower_tilde = self.decoder(z_e)
        return z_e_lower_tilde, z_e_lower, z_e

    def on_train_start(self):
        #cosList = [0.1,0.05,0.01,0.005,0.0001]
        cosList = [0,0,0,0,0]
        self.Cos_coeff = cosList[self.layer]
        mlflow.log_params({
            "input_feat_dim": self.input_feat_dim,
            "codebook_dim": self.codebook_dim,
            "enc_hidden_dim": self.enc_hidden_dim,
            "dec_hidden_dim": self.dec_hidden_dim,
            "num_res_blocks": self.num_res_blocks,
            "lr": self.lr,
            "decay": self.decay,
            "clip_grads": self.clip_grads,
            "layer": self.layer,
            "Cos_coeff": self.Cos_coeff,
            "batch_norm": self.batch_norm,
            "cos_reset": self.cos_reset,
            "compress": self.compress
        })

    def on_fit_end(self):
        mlflow.end_run()
        
    def cos_loss(self, original, reconstruction):
        cos_loss=torch.max(
                        1 - F.cosine_similarity(original, reconstruction, dim = 1),
                        torch.zeros(original.shape[0], original.shape[2], device=self.device)
                ).sum(dim=1).mean()
        return cos_loss

    def get_training_loss(self, x):
        recon, z_e_lower, z_e = self(x)
        recon_loss = self.recon_loss(z_e_lower, recon)
        cos_loss = self.cos_loss(z_e_lower, recon)
        dims = np.prod(recon.shape[1:]) # orig_w * orig_h * num_channels
        loss = recon_loss/dims 
        if not self.cos_reset or len(self) == 1:
            # Cosine reset is off OR Cosine reset is on and we are training layer 0
            loss += self.Cos_coeff*cos_loss/dims
        return cos_loss, recon_loss, loss

    

    def get_validation_loss(self, x):
        recon, z_e_lower, z_e = self(x, soft=False)
        recon_loss = self.recon_loss(z_e_lower, recon)
        val_cos_loss = self.cos_loss(z_e_lower, recon)
        dims = np.prod(recon.shape[1:]) # orig_w * orig_h * num_channels
        loss = recon_loss/dims
        if not self.cos_reset or len(self) == 1:
            # Cosine reset is off OR Cosine reset is on and we are training layer 0
            loss += self.Cos_coeff*val_cos_loss/dims
        return val_cos_loss, recon_loss, loss  



    def recon_loss(self, orig, recon):
        return F.mse_loss(orig, recon, reduction='none').sum(dim=(1,2)).mean()

    def decay_temp_linear(self, step, total_steps, temp_base, temp_min=0.001):
        factor = 1.0 - (step/total_steps)
        return temp_min + (temp_base - temp_min) * factor


    def training_step(self, batch, batch_idx):
        x, _ = batch
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        cos_loss, recon_loss, loss = self.get_training_loss(x)
        
        optimizer.zero_grad()
        self.manual_backward(loss)
        if self.clip_grads:
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("cos_loss", cos_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("recon", recon_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.train_outputs.update({"loss": loss, "cos_loss": cos_loss, "recon": recon_loss})

        return {"loss": loss, "cos_loss": cos_loss, "recon": recon_loss}

    def on_train_epoch_end(self):
        avg_loss = torch.mean(self.train_outputs['loss'])
        avg_cos_loss = torch.mean(self.train_outputs['cos_loss'])
        avg_recon = torch.mean(self.train_outputs['recon'])
        
        self.log("avg_loss", avg_loss, prog_bar=True)
        self.log("avg_cos_loss", avg_cos_loss, prog_bar=True)
        self.log("avg_recon", avg_recon, prog_bar=True)
        
        mlflow.log_metric("avg_loss", avg_loss.item(), step=self.current_epoch)
        mlflow.log_metric("avg_cos_loss", avg_cos_loss.item(), step=self.current_epoch)
        mlflow.log_metric("avg_recon", avg_recon.item(), step=self.current_epoch)

    def validation_step(self, val_batch, batch_idx):
        x, _ = val_batch
        cos_loss, recon_loss, loss = self.get_validation_loss(x)
        
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
        self.log("val_cos_loss", cos_loss, prog_bar=False, sync_dist=True, on_step=True, on_epoch=True)
        self.log("val_recon", recon_loss, prog_bar=False, sync_dist=True, on_step=True, on_epoch=True)
        self.val_outputs.update({"val_loss": loss, "val_cos_loss": cos_loss, "val_recon": recon_loss})
        return {"val_loss": loss, "val_cos_loss": cos_loss, "val_recon": recon_loss}
    
    def on_validation_epoch_end(self):
        avg_loss = torch.mean(self.val_outputs['val_loss'])
        avg_cos_loss = torch.mean(self.val_outputs['val_cos_loss'])
        avg_recon = torch.mean(self.val_outputs['val_recon'])
        
        self.log("avg_val_loss", avg_loss, prog_bar=True, sync_dist=True)
        self.log("avg_val_cos_loss", avg_cos_loss, prog_bar=True, sync_dist=True)
        self.log("avg_val_recon", avg_recon, prog_bar=True, sync_dist=True)
        
        mlflow.log_metric("avg_val_loss", avg_loss.item(), step=self.current_epoch)
        mlflow.log_metric("avg_val_cos_loss", avg_cos_loss.item(), step=self.current_epoch)
        mlflow.log_metric("avg_val_recon", avg_recon.item(), step=self.current_epoch)

    

    def test_step(self, test_batch, batch_idx):

        x,_ = test_batch
        cos_loss, recon_loss, loss = self.get_validation_loss(x)

        self.log("tst_loss", loss, prog_bar=False)
        self.log("tst_cos_loss", cos_loss, prog_bar=False)
        self.log("tst_recon", recon_loss, prog_bar=False)

        return loss    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=4e-4)
        lr_scheduler = FlatCA(optimizer, steps=1, eta_min=4e-5)

        return [optimizer], [lr_scheduler]

    def encode_lower(self, x):
        if self.prev_model is None:
            return x
        else:
            with torch.no_grad():
                z_e_lower = self.prev_model.encode(x)
                z_e_lower = self.normalize(z_e_lower)
            return z_e_lower

    def encode(self, x):
        with torch.no_grad():
            z_e_lower = self.encode_lower(x)
            z_e = self.encoder(z_e_lower)
        return z_e


    def decode_lower(self, z_q_lower):
        with torch.no_grad():
            recon = self.prev_model.decode(z_q_lower)           
        return recon

    def decode(self, z_q):
        with torch.no_grad():
            if self.prev_model is not None:
                z_e_u = self.normalize.unnorm(self.decoder(z_q))
                z_q_lower_tilde = self.prev_model.quantize(z_e_u)
                recon = self.decode_lower(z_q_lower_tilde)
            else:
                recon = self.decoder(z_q)
        return recon

    def quantize(self, z_e):
        return z_e


    def reconstruct_average(self, x, num_samples=10):
        """Average over stochastic edecodes"""

        b, c, h = x.shape
        result = torch.empty((num_samples, b, c, h))#.to(device)

        for i in range(num_samples):
            result[i] = self.decode(self.quantize(self.encode(x)))

        return result.mean(0)

    def reconstruct(self, x):
        return self.decode(self.quantize(self.encode(x)))


    def reconstruct_from_z_e(self, z_e):
        return self.decode(self.quantize(z_e))

    
    def __len__(self):
        i = 1
        layer = self
        while layer.prev_model is not None:
            i += 1
            layer = layer.prev_model
        return i

    def __getitem__(self, idx):
        max_layer = len(self) - 1
        if idx > max_layer:
            raise IndexError("layer does not exist")
        layer = self
        for _ in range(max_layer - idx):
            layer = layer.prev_model
        return layer

    def parameters(self, prefix="", recurse=True):
        for module in [self.encoder, self.decoder]:
            for name, param in module.named_parameters(recurse=recurse):
                yield param


    @classmethod
    def init_higher(cls, prev_model, **kwargs):
        model = HAE(prev_model.codebook_dim, prev_model=prev_model, **kwargs)
        model.prev_model.eval()
        return model

    @classmethod
    def init_bottom(cls, input_feat_dim, **kwargs):
        model = HAE(input_feat_dim,prev_model=None, **kwargs)
        return model
