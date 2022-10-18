from json import decoder
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import monai
from Models.MAE_video import MAEVideoModel
from utils import LinearWarmupCosineAnnealingLR, load_config, save_model
import data_setup_vid as data_setup
import os
import datetime
import sys

# SETTING UP CONFIGS

CONFIG_PATH = "configs/"
config_name = sys.argv[1]

config = load_config(CONFIG_PATH, config_name)
wandb_logger = WandbLogger(name=config["wandb"]["run_name"] + "-" + str(datetime.datetime.now()), project=config["wandb"]["project"], log_model="false")

# LIGHTNING MODEL
class MAEVideoLightning(pl.LightningModule):

    def __init__(self,
        # encoder params
        in_channels,
        img_size,
        patch_size,
        
        # reconstruction decoder params
        decoder_dim,
        masking_ratio,
        
        NUM_EPOCHS,
        LEARNING_RATE,
        WARMUP_EPOCHS,

        weight_decay=0.01,
        aug_dataloader=None):
        super().__init__()

        self.model = MAEVideoModel(in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            decoder_dim=decoder_dim,
            masking_ratio=masking_ratio)

        self.NUM_EPOCHS = NUM_EPOCHS
        self.LEARNING_RATE = LEARNING_RATE
        self.WARMUP_EPOCHS = WARMUP_EPOCHS
        self.LOSS_FN = monai.losses.DiceCELoss(smooth_nr=0, smooth_dr=1e-6)
        self.dice_metric = monai.metrics.DiceMetric()
        self.weight_decay = weight_decay
        self.aug_dataloader = aug_dataloader
        self.save_hyperparameters()
        self.Tanh = nn.Tanh()
        self.MSELoss = nn.MSELoss()

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

        # augmentation dataloader
        if(self.aug_dataloader is not None):
            X_aug, y_aug = next(iter(self.aug_dataloader))
            X_aug = X_aug.type(torch.cuda.FloatTensor)
            y_aug = y_aug.type(torch.cuda.FloatTensor)
            X = torch.cat((X, X_aug), 0)
            y = torch.cat((y, y_aug), 0)

        loss, reconstruction_output = self.model(X)
        self.log("loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        X = X.type(torch.cuda.FloatTensor)
        y = y.type(torch.cuda.FloatTensor)
        val_loss, reconstruction_output = self.model(X)
        self.log("val_loss", val_loss, sync_dist=True)
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

model = MAEVideoLightning(
    in_channels=config["model_parameters"]["in_channels"],
    img_size=(config["model_parameters"]["img_size_w"], config["model_parameters"]["img_size_h"]),
    patch_size=config["model_parameters"]["patch_size"],
    decoder_dim=config["model_parameters"]["decoder_dim"],
    masking_ratio=config["model_parameters"]["masking_ratio"],
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


trainer = pl.Trainer(devices=NO_GPUS, 
        accelerator="gpu", 
        strategy="ddp", 
        logger=wandb_logger, 
        callbacks=[overall_checkpoint_callback, val_loss_checkpoint_callback], 
        max_epochs=config["training_parameters"]["num_epochs"])

trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

