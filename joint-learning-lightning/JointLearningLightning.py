from json import decoder
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import monai
from SupportModels import JointModel
from utils import LinearWarmupCosineAnnealingLR
import data_setup
import os
import datetime
import wandb

wandb.init(
    project="joint-learning",
    config={
        "dataset": "NerveDataset",
        "model": "JointLearningModel",
        "learning_rate": 0.0001,
        "optimizer": "AdamW",
        "batch_size": 32,
        "loss_fn": "DiceCELoss",
    },
    name="JointLearning-NerveDataset-1000epochs-"+str(datetime.datetime.now())
)


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
        LAMBDA 
        ):
        super().__init__()
        self.model = JointModel(in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            decoder_dim=decoder_dim,
            masking_ratio=masking_ratio,
            out_channels=out_channels)
        self.LAMBDA = LAMBDA
        self.NUM_EPOCHS = 1000
        self.LOSS_FN = monai.losses.DiceCELoss(include_background=False, smooth_nr=0, smooth_dr=1e-6)
        self.OPTIMIZER = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.SCHEDULER = LinearWarmupCosineAnnealingLR(self.OPTIMIZER, warmup_epochs=50, max_epochs=self.NUM_EPOCHS)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        X = X.type(torch.cuda.FloatTensor)
        y = y.type(torch.cuda.FloatTensor)
        segmentation_pred, recon_score, recon_pred = self.model(X)
        loss = self.LOSS_FN(segmentation_pred, y) + recon_score
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        X = X.type(torch.cuda.FloatTensor)
        y = y.type(torch.cuda.FloatTensor)
        segmentation_pred, recon_score, recon_pred = self.model(X)
        val_loss = self.LOSS_FN(segmentation_pred, y) + recon_score
        return val_loss

    def training_epoch_end(self, outputs) -> None:
        gathered = self.all_gather(outputs)
        if self.global_rank == 0:
            loss = sum(output['loss'].mean() for output in gathered) / len(outputs)
            print(f"Epoch: {self.current_epoch} | loss: {loss.item()}")
            wandb.log({"train_loss": loss})

DATASET = "NerveDataset"
BATCH_SIZE = 8
train_dataloader, test_dataloader, val_dataloader = data_setup.create_dataloaders(
        dataset= os.path.join("../data", DATASET),
        batch_size=BATCH_SIZE
    )

IN_CHANNELS = 1
IMG_SIZE = 256
PATCH_SIZE = 16
DECODER_DIM = 768
MASKING_RATIO=  0.75
OUT_CHANNELS = 1
LAMBDA = 1


model = JointModelLightning(in_channels=IN_CHANNELS,
    img_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    decoder_dim=DECODER_DIM,
    masking_ratio=MASKING_RATIO,
    out_channels=OUT_CHANNELS,
    LAMBDA=LAMBDA)

NO_GPUS = torch.cuda.device_count()

trainer = pl.Trainer(devices=NO_GPUS, accelerator="gpu", strategy="ddp", max_epochs=1000)
trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
wandb.finish()
