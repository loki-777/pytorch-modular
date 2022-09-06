from json import decoder
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import monai
from SupportModels import JointModel
from utils import LinearWarmupCosineAnnealingLR
import data_setup
import os
import datetime
import wandb

config = {
        "DATASET": "NerveDataset",
        "MODEL": "JointLearningModel",
        "LEARNING_RATE": 0.0001,
        "WARMUP_EPOCHS": 50,
        "OPTIMIZER": "AdamW",
        "BATCH_SIZE": 8,
        "LOSS_FN": "DiceCELoss + LAMBDA*MSELoss",
        "IMG_SIZE": (320, 336),
        "IN_CHANNELS": 1,
        "PATCH_SIZE": 16,
        "DECODER_DIM": 768,
        "MASKING_RATIO": 0.75,
        "OUT_CHANNELS": 1,
        "LAMBDA": 1,
        "NUM_EPOCHS": 200,
        "SCHEDULER": "LinearWarmupCosineAnnealingLR"
        }

wandb.init(
    project="joint-learning",
    config=config,
    name="JointLearning-NerveDataset-200epochs-"+str(datetime.datetime.now())
)

resume = False


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
        WARMUP_EPOCHS
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
        return {"val_loss": val_loss, "val_segmentation_loss": segmentation_loss, "val_reconstruction_loss": recon_score}

    def validation_epoch_end(self, outputs) -> None:
        gathered = self.all_gather(outputs)
        if self.global_rank == 0:
            train_loss = sum(output['loss'].mean() for output in gathered) / len(outputs)
            train_segmentation_loss = sum(output['train_segmentation_loss'].mean() for output in gathered) / len(outputs)
            train_reconstruction_loss = sum(output['train_reconstruction_loss'].mean() for output in gathered) / len(outputs)
            val_loss = sum(output['val_loss'].mean() for output in gathered) / len(outputs)
            val_segmentation_loss = sum(output['val_segmentation_loss'].mean() for output in gathered) / len(outputs)
            val_reconstruction_loss = sum(output['val_reconstruction_loss'].mean() for output in gathered) / len(outputs)
            print(f"Epoch: {self.current_epoch} | loss: {train_loss.item()} | train_segmentation_loss: {train_segmentation_loss.item()} | train_reconstruction_loss: {train_reconstruction_loss.item()} | val_loss: {val_loss.item()} |  val_segmentation_loss: {val_segmentation_loss.item()} |  val_reconstruction_loss: {val_reconstruction_loss.item()}")
            wandb.log({"train_loss": train_loss, "train_segmentation_loss": train_segmentation_loss, "train_reconstruction_loss": train_reconstruction_loss, "val_loss": val_loss, "val_segmentation_loss": val_segmentation_loss, "val_reconstruction_loss": val_reconstruction_loss})






train_dataloader, test_dataloader, val_dataloader = data_setup.create_dataloaders(
        dataset= os.path.join("../data", config["DATASET"]),
        batch_size=config["BATCH_SIZE"],
        img_size=config["IMG_SIZE"]
    )




model = JointModelLightning(in_channels=config["IN_CHANNELS"],
    img_size=config["IMG_SIZE"],
    patch_size=config["PATCH_SIZE"],
    decoder_dim=config["DECODER_DIM"],
    masking_ratio=config["MASKING_RATIO"],
    out_channels=config["OUT_CHANNELS"],
    LAMBDA=config["LAMBDA"],
    NUM_EPOCHS=config["NUM_EPOCHS"],
    LEARNING_RATE=config["LEARNING_RATE"],
    WARMUP_EPOCHS=config["WARMUP_EPOCHS"]
    )




NO_GPUS = torch.cuda.device_count()




checkpoint_callback = ModelCheckpoint(dirpath="./checkpoints",save_last=True, every_n_epochs=20, save_on_train_epoch_end = True, filename='{epoch}-{train_loss:.4f}')
trainer = pl.Trainer(devices=NO_GPUS, accelerator="gpu", strategy="ddp", max_epochs=config["NUM_EPOCHS"], callbacks=[checkpoint_callback])

if(resume == True):
    trainer = pl.Trainer(devices=NO_GPUS, accelerator="gpu", strategy="ddp", max_epochs=config["NUM_EPOCHS"], resume_from_checkpoint='./checkpoints/last.ckpt', callbacks=[checkpoint_callback])

trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
wandb.finish()
