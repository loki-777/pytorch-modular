import os
from turtle import pos
import torch
import data_setup, engine, models, utils
import sys
import wandb
import datetime
import torch.nn as nn
import monai
from utils import LinearWarmupCosineAnnealingLR

CONFIG_PATH = "../configs/"
config_name = sys.argv[1]

config = utils.load_config(CONFIG_PATH, config_name)

# model and dset
DATASET = config["dataset_name"]
MODEL = config["model_name"]
DEVICE = config["device"]

# training params
NUM_EPOCHS = config["training_parameters"]["num_epochs"]
BATCH_SIZE = config["training_parameters"]["batch_size"]
LEARNING_RATE = config["training_parameters"]["learning_rate"]
OPTIMIZER = config["training_parameters"]["optimizer"]
LOSS_FN = config["training_parameters"]["loss_fn"]
ACTIVATION = config["training_parameters"]["activation"]
DROPOUT = config["training_parameters"]["dropout"]
TRAIN_VAL_SPLIT = config["training_parameters"]["train_val_split"]
LAMBDA = config["training_parameters"]["lambda"]

# testing params
MODEL_PTH = config["testing_parameters"]["model_path"]
TEST_LOSS_FN = config["testing_parameters"]["loss_fn"]
METRIC = config["testing_parameters"]["metric"]
SAVE = config["testing_parameters"]["save"]
DATA = config["testing_parameters"]["data"]

# save model
SAVE_MODEL_NAME = config["save_model"]["model_name"]

if config["log"] == True:
    wandb.init(
        project=config["wandb"]["project"],
        config={
            "dataset": DATASET,
            "model": MODEL,
            "learning_rate": LEARNING_RATE,
            "optimizer": OPTIMIZER,
            "batch_size": BATCH_SIZE,
            "loss_fn": LOSS_FN,
            "train_size": 600,
            "val_size": 200
        },
        name=config["wandb"]["name"]+str(datetime.datetime.now())
    )

WANDB_PARAMS = {
    "log": config["log"],
    "fields": config["wandb"]["fields"]
}

print(f"-------{MODEL} | {DATASET} | {DEVICE}-------")
if __name__ ==  '__main__':
    train_dataloader, test_dataloader, val_dataloader = data_setup.create_dataloaders(
        dataset= os.path.join("../data", DATASET),
        batch_size=BATCH_SIZE
    )
    model = models.JointModel(in_channels=config["model_parameters"]["in_channels"], 
    out_channels=config["model_parameters"]["out_channels"], 
    img_size=config["model_parameters"]["img_size"], 
    patch_size=config["model_parameters"]["patch_size"],
    decoder_dim=config["model_parameters"]["decoder_dim"], 
    masking_ratio=config["model_parameters"]["masking_ratio"]).to(DEVICE)
    # model = models.UNETR(in_channels=1, out_channels=1, img_size=256, feature_size=32, norm_name='batch', spatial_dims=2).to(DEVICE)
    if config["test_or_train"]:
        print("...TRAINING...")
        LOSS_FN = monai.losses.DiceCELoss(include_background=False, smooth_nr=0, smooth_dr=1e-6)
        OPTIMIZER = torch.optim.AdamW(model.parameters(), lr=1e-4)
        SCHEDULER = LinearWarmupCosineAnnealingLR(OPTIMIZER, warmup_epochs=50, max_epochs=NUM_EPOCHS)
        train_results = engine.train(model=model,
                    train_dataloader=train_dataloader,
                    val_dataloader=test_dataloader,
                    loss_fn=LOSS_FN,
                    optimizer=OPTIMIZER,
                    scheduler=SCHEDULER,
                    epochs=NUM_EPOCHS,
                    LAMBDA=LAMBDA,
                    wandb_params=WANDB_PARAMS,
                    device=DEVICE)

        if config["save_model"]["save"]:
            utils.save_model(model=model, target_dir="saved_models", model_name=SAVE_MODEL_NAME)

    else:
        print("...TESTING...")
        model.load_state_dict(torch.load(MODEL_PTH))
        LOSS_FN = monai.losses.DiceCELoss(include_background=False, smooth_nr=0, smooth_dr=1e-6)
        if DATA == "test":
            test_results = engine.test(model=model, test_dataloader=test_dataloader, loss_fn=LOSS_FN, device=DEVICE, save=SAVE, data=DATA)
        else:
            test_results = engine.test(model=model, test_dataloader=val_dataloader, loss_fn=LOSS_FN, device=DEVICE, save=SAVE, data=DATA)

    if config["log"] == True:
        wandb.finish()

