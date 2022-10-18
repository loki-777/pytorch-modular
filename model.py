from json import decoder
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import monai
from Models import JointModel
from utils import LinearWarmupCosineAnnealingLR, load_config, save_model
import data_setup
import os
import datetime
import sys
import GPUtil

# LIGHTNING MODEL
class JointModelLightning(pl.LightningModule):

    def __init__(self,
        # encoder params
        in_channels,
        img_size,
        patch_size,
        
        # reconstruction params
        decoder_dim,
        masking_ratio,
        
        # segmentation decoder params
        out_channels,
        
        #loss function params
        LAMBDA,

        NUM_EPOCHS,
        LEARNING_RATE,
        WARMUP_EPOCHS, 
        WEIGHT_DECAY
        ):
        super().__init__()

        self.model = JointModel(in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            decoder_dim=decoder_dim,
            masking_ratio=masking_ratio,
            out_channels=out_channels)

        self.LAMBDA = LAMBDA
        self.WEIGHT_DECAY = WEIGHT_DECAY
        self.NUM_EPOCHS = NUM_EPOCHS
        self.LEARNING_RATE = LEARNING_RATE
        self.WARMUP_EPOCHS = WARMUP_EPOCHS
        self.LOSS_FN = monai.losses.DiceCELoss(smooth_nr=0, smooth_dr=1e-6)
        self.OPTIMIZER = torch.optim.AdamW(self.model.parameters(), lr=self.LEARNING_RATE)
        self.SCHEDULER = LinearWarmupCosineAnnealingLR(self.OPTIMIZER, warmup_epochs=self.WARMUP_EPOCHS, max_epochs=self.NUM_EPOCHS)
        self.dice_metric = monai.metrics.DiceMetric()
	self.save_hyperparameters()

    def forward(self, X):
        return self.model(X)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.LEARNING_RATE, weight_decay = self.WEIGHT_DECAY)
        scheduler =  LinearWarmupCosineAnnealingLR(self.OPTIMIZER, warmup_epochs=self.WARMUP_EPOCHS, max_epochs=self.NUM_EPOCHS)
        return [optimizer, scheduler]

    def training_step(self, batch, batch_idx):
        X, y = batch
        X = X.type(torch.cuda.FloatTensor)
        y = y.type(torch.cuda.FloatTensor)
        segmentation_pred, recon_score, recon_pred = self.model(X)
        segmentation_loss = self.LOSS_FN(segmentation_pred, y)
        loss = segmentation_loss + self.LAMBDA*recon_score
        self.log("loss", loss, sync_dist=True)
        self.log("segmentation_loss", segmentation_loss, sync_dist=True)
        self.log("reconstruction_loss", recon_score, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        X = X.type(torch.cuda.FloatTensor)
        y = y.type(torch.cuda.FloatTensor)
        segmentation_pred, recon_score, recon_pred = self.model(X)
        segmentation_loss = self.LOSS_FN(segmentation_pred, y)
        val_loss = segmentation_loss + self.LAMBDA*recon_score
        segmentation_pred = (segmentation_pred>0.5)
        y = y > 0.5
        self.dice_metric(y_pred=segmentation_pred, y=y)
        val_segmentation_accuracy = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        self.log("val_loss", val_loss, sync_dist=True)
        self.log("val_segmentation_loss", segmentation_loss, sync_dist=True)
        self.log("val_reconstruction_loss", recon_score, sync_dist=True)
        self.log("val_acc", val_segmentation_accuracy, sync_dist=True)
        return val_loss
