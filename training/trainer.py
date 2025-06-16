from training.perceiver import *
from training.atomiser import *
from training.utils import *
from training.losses import *
from training.VIT import *
from training.ScaleMae import*
from training.Unet import*
from training.ResNet import *
from collections import defaultdict
from training import *
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
import einops as einops
from einops import rearrange, repeat
from einops.layers.torch import Reduce
import matplotlib.pyplot as plt

import random
import torchmetrics
import warnings
import wandb
from transformers import get_cosine_schedule_with_warmup
from torchmetrics.segmentation import DiceScore

#BigEarthNet...
warnings.filterwarnings("ignore", message="No positive samples found in target, recall is undefined. Setting recall to one for all thresholds.")

class Model(pl.LightningModule):
    def __init__(self, config, wand,transform):
        super().__init__()
        self.strict_loading = False
        self.config = config
        self.transform=transform
        self.wand = wand
        self.num_classes = config["trainer"]["num_classes"]
        self.logging_step = config["trainer"]["logging_step"]
        self.actual_epoch = 0
        
        self.weight_decay = float(config["trainer"]["weight_decay"])
        self.mode = "training"
        self.multi_modal = config["trainer"]["multi_modal"]
        self.table=False
        self.comment_log=""

        
        self.metric_IoU = torchmetrics.JaccardIndex(task="multiclass", num_classes=self.num_classes, average="macro")
        self.metric_Dice= DiceScore(num_classes=4, average='macro', input_format='index')
        self.metric_Precision =  torchmetrics.Precision(task="multiclass", num_classes=4, average='macro')
        self.metric_Recall =  torchmetrics.Recall(task="multiclass", num_classes=4, average='macro')

        
        
        self.tmp_val_loss = 0
        self.tmp_val_ap = 0

        if config["encoder"] == "Unet":
            self.encoder = UNet(num_classes=4,
                                 in_channels=12, 
                                 depth=5, 
                                 start_filts=64, 
                                 up_mode='transpose', 
                                 merge_mode='concat')



        if config["encoder"] == "Atomiser":
            self.encoder = Atomiser(
                config=self.config,
                transform=self.transform,
                depth=config["Atomiser"]["depth"],
                num_latents=config["Atomiser"]["num_latents"],
                latent_dim=config["Atomiser"]["latent_dim"],
                cross_heads=config["Atomiser"]["cross_heads"],
                latent_heads=config["Atomiser"]["latent_heads"],
                cross_dim_head=config["Atomiser"]["cross_dim_head"],
                latent_dim_head=config["Atomiser"]["latent_dim_head"],
                num_classes=config["trainer"]["num_classes"],
                attn_dropout=config["Atomiser"]["attn_dropout"],
                ff_dropout=config["Atomiser"]["ff_dropout"],
                weight_tie_layers=config["Atomiser"]["weight_tie_layers"],
                self_per_cross_attn=config["Atomiser"]["self_per_cross_attn"],
                final_classifier_head=config["Atomiser"]["final_classifier_head"],
                masking=config["Atomiser"]["masking"]
            )


        self.loss = nn.CrossEntropyLoss()
        self.lr = float(config["trainer"]["lr"])
        
    def forward(self, x,mask=None,resolution=None,training=True):
        if "Atomiser" in self.config["encoder"]:
            return self.encoder(x,mask,resolution,training=training)
        else:
            return self.encoder(x)
                
            
    def training_step(self, batch, batch_idx):
        x,y,resolution= batch
        y_hat = self.forward(x,resolution=resolution,training=True)
        preds = torch.argmax(y_hat.clone(), dim=1)
        loss = self.loss(y_hat, y)

        

        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=False, sync_dist=False)

        return loss
    

    def update_metrics(self,preds,y):

        self.metric_IoU.update(preds,y)
        self.metric_Dice.update(preds,y)
        self.metric_Precision.update(preds,y)
        self.metric_Recall.update(preds,y)

    

    def compute_metrics(self):

        IoU=self.metric_IoU.compute()
        Dice=self.metric_Dice.compute()
        Precision=self.metric_Precision.compute()
        Recall=self.metric_Recall.compute()
        return IoU,Dice,Precision,Recall
    
    def reset_metrics(self):

        self.metric_IoU.reset()
        self.metric_Dice.reset()
        self.metric_Precision.reset()
        self.metric_Recall.reset()
    
    
    
        
    def on_train_epoch_end(self):

        metrics = self.trainer.callback_metrics
        train_loss = metrics.get("train_loss", float("inf"))
        
       
        self.log("train_loss", train_loss, on_step=False, on_epoch=True, logger=True)
        self.log("log train_loss", np.log(train_loss.item()+ 1e-8), on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
        return {"train_loss": train_loss}
    
    #def on_train_batch_end(self, outputs, batch, batch_idx):
    #    # only on the very first real training batch, and only on rank 0
    #    if batch_idx == 0 and self.trainer.global_rank == 0:
    #        missing = []
    #        for name, p in self.named_parameters():
    #            if p.requires_grad and p.grad is None:
    #                missing.append(name)
    #        if missing:
    #            print("⚠️ Still no grad for:\n" + "\n".join(missing))
    #        else:
    #            print("✅ All parameters got gradients!")
    



        
    def validation_step(self, batch, batch_idx,dataloader_idx=0):
        x,y,resolution= batch
        y_hat = self.forward(x,resolution=resolution,training=True)
        preds = torch.argmax(y_hat.clone(), dim=1)
        loss = self.loss(y_hat, y)


        self.update_metrics(preds,y)

        

        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=False, sync_dist=False)

        return loss   

    def on_validation_epoch_end(self):

        metrics = self.trainer.callback_metrics
        val_loss = metrics.get("val_loss", float("inf"))
        
        IoU,Dice,Precision,Recall=self.compute_metrics()
       
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, logger=True)
        self.log("log val_loss", np.log(val_loss.item()+ 1e-8), on_step=False, on_epoch=True, logger=True, sync_dist=True)

        self.log("val_IoU", IoU, on_step=False, on_epoch=True, logger=True)
        self.log("val_Dice",Dice, on_step=False, on_epoch=True, logger=True)
        self.log("val_Precision", Precision, on_step=False, on_epoch=True, logger=True)
        self.log("val_Recall", Recall, on_step=False, on_epoch=True, logger=True)

        self.reset_metrics()

        
        return {"val_loss": val_loss}
        
        
        
    
        
        
    def test_step(self, batch, batch_idx):
        x,y,resolution= batch
        y_hat = self.forward(x,resolution=resolution,training=True)
        preds = torch.argmax(y_hat.clone(), dim=1)
        loss = self.loss(y_hat, y)


        self.update_metrics(preds,y)

        return loss  

    def on_test_epoch_end(self):
        IoU,Dice,Precision,Recall=self.compute_metrics()
       
      
        self.log("test_IoU", IoU, on_step=False, on_epoch=True, logger=True)
        self.log("test_Dice",Dice, on_step=False, on_epoch=True, logger=True)
        self.log("test_Precision", Precision, on_step=False, on_epoch=True, logger=True)
        self.log("test_Recall", Recall, on_step=False, on_epoch=True, logger=True)

        self.reset_metrics()

        
        
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        accumulate_grad_batches = 64#self.config["trainer"].get("accumulate_grad_batches", 1)
        batches_per_epoch = self.trainer.estimated_stepping_batches/self.config["trainer"]["epochs"]
        steps_per_epoch = batches_per_epoch // accumulate_grad_batches

        total_steps = self.config["trainer"]["epochs"] * steps_per_epoch
        warmup_steps = int(0.1 * total_steps)  # 10% warmup

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # step-wise updating
                'monitor': 'val_mod_val_loss'
            }
        }