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

# model params
NUM_CLASSES = config["model_parameters"]["num_classes"]
IN_CHANNELS = config["model_parameters"]["in_channels"]
DEPTH = config["model_parameters"]["depth"]
SEED_FILTERS = config["model_parameters"]["seed_filters"]

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
    model = models.UNet(
        num_classes=NUM_CLASSES,
        in_channels=IN_CHANNELS,
        depth=DEPTH,
        seed_filters=SEED_FILTERS).to(DEVICE)
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