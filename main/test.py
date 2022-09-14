import torch
import os
import wandb
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import cv2 as cv
import numpy as np
from sklearn.metrics import f1_score
from train import JointModelLightning

def test_step(model: torch.nn.Module, 
              img_input_dir,
              mask_input_dir,
              img_size,
              output_dir,
              threshold,
              device: torch.device,
              save) -> Tuple[float, float]:
     
    test_loss = 0

    model.to(device)
    model.eval()

    print(model)

    if(not os.path.exists(output_dir)):
        os.mkdir(output_dir)
        os.mkdir(os.path.join(output_dir,'reconstruction'))
        os.mkdir(os.path.join(output_dir,'masks'))

    img_path_list = os.listdir(img_input_dir)

    test_loss = 0
    overall_recon_loss = 0

    with torch.inference_mode():
        for img_path in img_path_list:
            X = cv.imread(os.path.join(img_input_dir, img_path),0)
            y = cv.imread(os.path.join(mask_input_dir, img_path),0)
            X = cv.resize(X, (img_size[1], img_size[0]), interpolation = cv.INTER_NEAREST)
            y = cv.resize(y, (img_size[1], img_size[0]), interpolation = cv.INTER_NEAREST)
            X = np.expand_dims(X, 0)/255
            y = np.expand_dims(y, 0)/255
            X = torch.Tensor(X)
            y = torch.Tensor(y)
            X, y = X.to(device), y.to(device)
            X = X.type(torch.cuda.FloatTensor)
            y = y.type(torch.cuda.FloatTensor)
            segmentation_pred, recon_loss, recon_pred = model(X)
            segmentation_pred = (segmentation_pred > threshold) + 0 
            segmentation_pred = torch.squeeze(segmentation_pred).cpu().numpy()
            recon_pred = torch.squeeze(recon_pred).cpu().numpy()
            y = torch.squeeze(y).cpu().numpy()

            test_loss += f1_score(y, segmentation_pred)
            overall_recon_loss += recon_loss

            if(save):
                segmentation_pred *= 255
                recon_pred *= 255

                cv.imwrite(os.path.join(output_dir,'reconstruction'), recon_pred)
                cv.imwrite(os.path.join(output_dir,'masks'), segmentation_pred)
    
    return test_loss / len(img_path_list), recon_loss / len(img_path_list)

def test(model: torch.nn.Module,
         img_input_dir,
         mask_input_dir,
         img_size,
         output_dir,
         threshold,
         device: torch.device,
         save: bool) -> Dict[str, List]:

    test_loss = test_step(model=model,
        img_input_dir=img_input_dir,
        mask_input_dir=mask_input_dir,
        img_size=img_size,
        output_dir=output_dir,
        threshold=threshold,
        device=device,
        save=save)
    print(
        f"mean_test_loss: {test_loss[0]:.4f}" + "|" + f"mean_reconstruction_loss: {test_loss[1]:.4f}"
    )

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

model = JointModelLightning.load_from_checkpoint(
    "checkpoints/epoch=199-step=259600.ckpt",
    in_channels=config["IN_CHANNELS"],
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

test(model=model, 
img_input_dir="../data/NerveDataset/test/images",
mask_input_dir="../data/NerveDataset/test/ground_truth",
img_size=(320, 336),
output_dir="../outputs",
threshold=0.5,
device="cuda:0",
save=True)
