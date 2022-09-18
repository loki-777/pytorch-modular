from json import decoder
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import monai
from Models import JointModel
from utils import LinearWarmupCosineAnnealingLR, load_config, save_model
import data_setup
import os
import datetime
import wandb, sys

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
        save,
        every_n_epoch
        ):
        super().__init__()
        
        self.model = JointModel(in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            decoder_dim=decoder_dim,
            masking_ratio=masking_ratio,
            out_channels=out_channels)

        self.LAMBDA = LAMBDA
        self.NUM_EPOCHS = NUM_EPOCHS
        self.LEARNING_RATE = LEARNING_RATE
        self.WARMUP_EPOCHS = WARMUP_EPOCHS
        self.LOSS_FN = monai.losses.DiceCELoss(include_background=False, smooth_nr=0, smooth_dr=1e-6)
        self.OPTIMIZER = torch.optim.AdamW(self.model.parameters(), lr=self.LEARNING_RATE)
        self.SCHEDULER = LinearWarmupCosineAnnealingLR(self.OPTIMIZER, warmup_epochs=self.WARMUP_EPOCHS, max_epochs=self.NUM_EPOCHS)
        self.dice_metric = monai.metrics.DiceMetric()

        self.global_min_validation_loss = 1000000
        self.global_max_validation_acc = 0
        self.save = save
        self.every_n_epoch = every_n_epoch
        self.best_val_loss_epoch = 0
        self.best_val_acc_epoch = 0
        self.save_hyperparameters()

    def forward(self, X):
        return self.model(X)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.LEARNING_RATE)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        X = X.type(torch.cuda.FloatTensor)
        y = y.type(torch.cuda.FloatTensor)
        segmentation_pred, recon_score, recon_pred = self.model(X)
        segmentation_loss = self.LOSS_FN(segmentation_pred, y)
        loss = segmentation_loss + self.LAMBDA*recon_score
        return {"loss": loss, "train_segmentation_loss": segmentation_loss, "train_reconstruction_loss": recon_score}
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        X = X.type(torch.cuda.FloatTensor)
        y = y.type(torch.cuda.FloatTensor)
        segmentation_pred, recon_score, recon_pred = self.model(X)
        segmentation_loss = self.LOSS_FN(segmentation_pred, y)
        val_loss = segmentation_loss + self.LAMBDA*recon_score
        self.dice_metric(y_pred=segmentation_pred, y=y)
        val_segmentation_accuracy = self.dice_metric.aggregate().item()
        self.dice_metric.reset()

        return {"val_loss": val_loss, "val_segmentation_loss": segmentation_loss, "val_reconstruction_loss": recon_score, "val_segmentation_accuracy": val_segmentation_accuracy}
    
    def training_epoch_end(self, outputs) -> None:
        gathered = self.all_gather(outputs)
        if self.global_rank == 0:
            train_loss = sum(output['loss'].mean() for output in gathered) / len(outputs)
            train_segmentation_loss = sum(output['train_segmentation_loss'].mean() for output in gathered) / len(outputs)
            train_reconstruction_loss = sum(output['train_reconstruction_loss'].mean() for output in gathered) / len(outputs)
            print(f"Epoch: {self.current_epoch} | loss: {train_loss.item()} | train_segmentation_loss: {train_segmentation_loss.item()} | train_reconstruction_loss: {train_reconstruction_loss.item()}")
            wandb.log({"train_epoch": self.current_epoch, "train_loss": train_loss, "train_segmentation_loss": train_segmentation_loss, "train_reconstruction_loss": train_reconstruction_loss})

    def validation_epoch_end(self, outputs) -> None:
        gathered = self.all_gather(outputs)
        if self.global_rank == 0:
            val_loss = sum(output['val_loss'].mean() for output in gathered) / len(outputs)
            val_segmentation_loss = sum(output['val_segmentation_loss'].mean() for output in gathered) / len(outputs)
            val_reconstruction_loss = sum(output['val_reconstruction_loss'].mean() for output in gathered) / len(outputs)
            val_segmentation_accuracy = sum(output['val_segmentation_accuracy'].mean() for output in gathered) / len(outputs)
            if(self.current_epoch % self.every_n_epoch == 0):
                if(self.save == True):
                    save_model(self.model, "saved_models/checkpoints", f"epoch-{self.current_epoch}.pth")
            if(val_loss < self.global_min_validation_loss):
                save_model(self.model, "saved_models", "best_val_loss.pth")
                self.global_min_validation_loss = val_loss
                self.best_val_loss_epoch = self.current_epoch

            if(val_segmentation_accuracy > self.global_max_validation_acc):
                save_model(self.model, "saved_models", "best_val_acc.pth")
                self.global_max_validation_acc = val_segmentation_accuracy
                self.best_val_acc_epoch = self.current_epoch
            
            print(f"Epoch: {self.current_epoch} | val_loss: {val_loss.item()} |  val_segmentation_loss: {val_segmentation_loss.item()} |  val_reconstruction_loss: {val_reconstruction_loss.item()}")
            wandb.log({"val_epoch": self.current_epoch, "val_loss": val_loss, "val_segmentation_loss": val_segmentation_loss, "val_reconstruction_loss": val_reconstruction_loss, "val_segmentation_accuracy": val_segmentation_accuracy})

