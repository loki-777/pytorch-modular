from json import decoder
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import monai
from SupportModels import JointModel
from utils import LinearWarmupCosineAnnealingLR, load_config
import data_setup
import os
import datetime
import wandb, sys

# SETTING UP CONFIGS

CONFIG_PATH = "configs/"
config_name = sys.argv[1]

config = load_config(CONFIG_PATH, config_name)


# WANDB INIT

wandb.init(
    project=config["wandb"]["project"],
    config=config["wandb"]["config"],
    name= config["wandb"]["run_name"] + "-" + str(datetime.datetime.now())
)


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



# DEFINE DATALOADERS

train_dataloader, test_dataloader, val_dataloader = data_setup.create_dataloaders(
        dataset= os.path.join("../data", config["dataset_name"]),
        batch_size=config["training_parameters"]["batch_size"],
        img_size=config["model_parameters"]["img_size"],
        train_transforms=config["training_parameters"]["transformation"]
    )


# INITIALISE MODEL

model = JointModelLightning(in_channels=config["IN_CHANNELS"],
    img_size=config["model_parameters"]["img_size"],
    patch_size=config["model_parameters"]["patch_size"],
    decoder_dim=config["model_parameters"]["decoder_dim"],
    masking_ratio=config["model_parameters"]["masking_ratio"],
    out_channels=config["model_parameters"]["out_channels"],
    LAMBDA=config["training_parameters"]["lambda"],
    NUM_EPOCHS=config["training_parameters"]["num_epochs"],
    LEARNING_RATE=config["training_parameters"]["learning_rate"],
    WARMUP_EPOCHS=config["training_parameters"]["warmup_epochs"]
    )

# TRAINING

NO_GPUS = torch.cuda.device_count()
resume = config["training_parameters"]["resum_training"]

checkpoint_callback = ModelCheckpoint(dirpath="./checkpoints",save_last=True, every_n_epochs=config["save_model"]["every_n_epoch"], save_on_train_epoch_end = True, filename='{epoch}-{train_loss:.4f}')

if(resume == True):
    trainer = pl.Trainer(devices=NO_GPUS, accelerator="gpu", strategy="ddp", max_epochs=config["training_parameters"]["num_epochs"], resume_from_checkpoint='./checkpoints/last.ckpt', callbacks=[checkpoint_callback])
else:
    trainer = pl.Trainer(devices=NO_GPUS, accelerator="gpu", strategy="ddp", max_epochs=config["training_parameters"]["num_epochs"], callbacks=[checkpoint_callback])

trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

wandb.finish()
