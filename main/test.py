import torch
import torch.nn as nn
import os
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import cv2 as cv
import numpy as np
from sklearn.metrics import f1_score
from model import JointModelLightning
from PIL import Image 
import PIL 

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
            y = np.zeros(X.shape, dtype='uint8')
            if(os.path.exists(os.path.join(mask_input_dir, img_path))):
                y = cv.imread(os.path.join(mask_input_dir, img_path),0)
            X = cv.resize(X, (img_size[1], img_size[0]), interpolation = cv.INTER_NEAREST)
            y = cv.resize(y, (img_size[1], img_size[0]), interpolation = cv.INTER_NEAREST)
            X = np.expand_dims(X, (0,1))/255
            y = np.expand_dims(y, (0,1))/255
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
            y = (y > 0) + 0
            y_f = y.flatten()
            segmentation_pred_f = segmentation_pred.flatten()
            print(np.unique(y))
            print(np.unique(segmentation_pred))

            test_loss += f1_score(y_f, segmentation_pred_f)
            overall_recon_loss += recon_loss

            if(save):
                segmentation_pred *= 255
                recon_pred *= 255

                recon_pred=Image.fromarray(recon_pred)
                segmentation_pred=Image.fromarray(segmentation_pred)

                recon_pred.save(os.path.join(output_dir,'reconstruction',img_path[:-3],'.png'), dpi=(300, 300))
                segmentation_pred.save(os.path.join(output_dir,'masks',img_path[:-3],'.png'), dpi=(300, 300))
    
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

CONFIG_PATH = "configs/"
config_name = sys.argv[1]

config = load_config(CONFIG_PATH, config_name)

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
        WEIGHT_DECAY=config["training_parameters"]["weight_decay"]
        )

model.load_state_dict(torch.load("Baseline/best_val_acc.pth"), strict=False)

test(model=model, 
img_input_dir="../data/NerveDataset/train/images",
mask_input_dir="../data/NerveDataset/train/ground_truth",
img_size=(320, 336),
output_dir="../outputs",
threshold=0.5,
device="cuda:0",
save=True)
