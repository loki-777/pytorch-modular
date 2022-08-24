from turtle import pos
import torch
import data_setup, engine, models, utils
import sys
import wandb

CONFIG_PATH = "../configs/"
config_name = sys.argv[1]

config = utils.load_config(CONFIG_PATH, config_name)

# dirs
TRAIN_DIR = config["train_dir"]
TEST_DIR = config["test_dir"]
MODEL_DIR = config["model_dir"]

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

# testing params
MODEL_PTH = config["testing_parameters"]["model_path"]
TEST_LOSS_FN = config["testing_parameters"]["loss_fn"]
METRIC = config["testing_parameters"]["metric"]

# save model
SAVE_MODEL_NAME = config["save_model"]["save_path"]

if config["log"] == True:
    wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        config=config["wandb"]["config"],
    )


train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=TRAIN_DIR,
    test_dir=TEST_DIR,
    batch_size=BATCH_SIZE
)

WANDB_PARAMS = {
    "log": config["log"],
    "fields": config["wandb"]["fields"]
}

if config["test_or_train"]:
    model = models.UNETR(
        in_channels=config["model_parameters"]["in_channels"],
        out_channels=config["model_parameters"]["out_channels"],
        img_size=config["model_parameters"]["img_size"],
        feature_size=config["model_parameters"]["feature_size"],
        hidden_size=config["model_parameters"]["hidden_size"],
        mlp_dim=config["model_parameters"]["mlp_dim"],
        num_heads=config["model_parameters"]["num_heads"],
        pos_embed=config["model_parameters"]["pos_embed"],
        norm_name=config["model_parameters"]["norm_name"],
        conv_block=config["model_parameters"]["conv_block"],
        res_block=config["model_parameters"]["res_block"],
        dropout_rate=config["model_parameters"]["dropout_rate"],
        spatial_dims=config["model_parameters"]["spatial_dims"]
    ).to(DEVICE)
    
    train_results = engine.train(model=model,
                train_dataloader=train_dataloader,
                val_dataloader=test_dataloader,
                loss_fn=LOSS_FN,
                optimizer=OPTIMIZER,
                epochs=NUM_EPOCHS,
                wandb_params=WANDB_PARAMS,
                device=DEVICE)

    if config["save_model"]["save"]:
        utils.save_model(model=model, target_dir="saved_models", model_name=SAVE_MODEL_NAME)

else:
    model = torch.load(MODEL_PTH)
    test_results = engine.test(model=model, test_dataloader=test_dataloader, loss_fn=TEST_LOSS_FN, device=DEVICE)

if config["log"] == True:
    wandb.finish()