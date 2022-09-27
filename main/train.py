from json import decoder
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
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

# SETTING UP CONFIGS

CONFIG_PATH = "configs/"
config_name = sys.argv[1]

config = load_config(CONFIG_PATH, config_name)
wandb_logger = WandbLogger(name=config["wandb"]["run_name"] + "-" + str(datetime.datetime.now()), project=config["wandb"]["project"], log_model="false")

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

        weight_decay=0.01,
        aug_dataloader=None):
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
        self.LOSS_FN = monai.losses.DiceCELoss(smooth_nr=0, smooth_dr=1e-6)
        self.dice_metric = monai.metrics.DiceMetric()
        self.weight_decay = weight_decay
        self.aug_dataloader = aug_dataloader
        self.save_hyperparameters()

    def forward(self, X):
        return self.model(X)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.LEARNING_RATE, weight_decay=self.weight_decay)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.WARMUP_EPOCHS, max_epochs=self.NUM_EPOCHS)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        X, y = batch
        X = X.type(torch.cuda.FloatTensor)
        y = y.type(torch.cuda.FloatTensor)
        print("before", X.shape, y.shape)

        # augmentation dataloader
        if(self.aug_dataloader is not None):
            X_aug, y_aug = next(iter(self.aug_dataloader))
            X_aug = X_aug.type(torch.cuda.FloatTensor)
            y_aug = y_aug.type(torch.cuda.FloatTensor)
            X = torch.cat((X, X_aug), 0)
            y = torch.cat((y, y_aug), 0)

        print("after", X.shape, y.shape)
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



NO_GPUS = torch.cuda.device_count()


# DEFINE DATALOADERS

train_dataloader, test_dataloader, val_dataloader = data_setup.create_dataloaders(
        dataset= os.path.join("../data", config["dataset_name"]),
        batch_size=config["training_parameters"]["batch_size"],
        img_size=(config["model_parameters"]["img_size_w"], config["model_parameters"]["img_size_h"]),
        num_workers=8,
        augmentation=config["training_parameters"]["augmentation"])


# AUGMENTATION DATALOADER

aug_dataloader = None
if(config["training_parameters"]["augmentation"] == "addition"):
    aug_data = data_setup.CustomDatasetClass(os.path.join("../data", config["aug_dataset_name"]), (config["model_parameters"]["img_size_w"], config["model_parameters"]["img_size_h"]))
    aug_dataloader = DataLoader(
        aug_data,
        batch_size=config["training_parameters"]["aug_batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )



# INITIALISE MODEL

model = JointModelLightning(
    in_channels=config["model_parameters"]["in_channels"],
    img_size=(config["model_parameters"]["img_size_w"], config["model_parameters"]["img_size_h"]),
    patch_size=config["model_parameters"]["patch_size"],
    decoder_dim=config["model_parameters"]["decoder_dim"],
    masking_ratio=config["model_parameters"]["masking_ratio"],
    out_channels=config["model_parameters"]["out_channels"],
    LAMBDA=config["training_parameters"]["lambda"],
    NUM_EPOCHS=config["training_parameters"]["num_epochs"],
    LEARNING_RATE=config["training_parameters"]["learning_rate"],
    WARMUP_EPOCHS=config["training_parameters"]["warmup_epochs"],
    weight_decay=config["training_parameters"]["weight_decay"],
    aug_dataloader=aug_dataloader)



overall_checkpoint_callback = ModelCheckpoint(
        dirpath="saved_models/checkpoints/",
        filename="{epoch}",
        monitor="loss",
        save_top_k=-1,
        mode="min")

val_loss_checkpoint_callback = ModelCheckpoint(
        dirpath="saved_models",
        filename="best_val_loss-{epoch}",
        monitor="val_loss",
        save_top_k=1,
        mode="min")

val_acc_checkpoint_callback = ModelCheckpoint(
        dirpath="saved_models",
        filename="best_val_acc-{epoch}",
        monitor="val_acc",
        save_top_k=1,
        mode="max")


trainer = pl.Trainer(devices=NO_GPUS, 
        accelerator="gpu", 
        strategy="ddp", 
        logger=wandb_logger, 
        callbacks=[overall_checkpoint_callback, val_loss_checkpoint_callback, val_acc_checkpoint_callback], 
        max_epochs=config["training_parameters"]["num_epochs"])

trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

