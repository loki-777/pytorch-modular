import torch
import os
import wandb
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import cv2
import numpy as np

def test_step(model: torch.nn.Module, 
              img_input_dir,
              mask_input_dir,
              output_dir,
              loss_fn: torch.nn.Module,
              device: torch.device,
              save,
              data) -> Tuple[float, float]:
     
    test_loss = 0

    model.to(device)
    model.eval()

    print(model)

    if(not os.path.exists(output_dir)):
        os.mkdir(output_dir)
        os.mkdir(os.path.join(output_dir,'recons'))
        os.mkdir(os.path.join(output_dir,'masks'))

    img_path_list = os.listdir(img_input_dir)

    with torch.inference_mode():
        for img_path in img_path_list:
            X = cv2.imread(os.path.join(img_input_dir, img_path),0)
            y = cv2.imread(os.path.join(mask_input_dir, img_path),0)
            X = cv2.resize(X, (336, 320), interpolation = cv2.INTER_NEAREST)
            y = cv2.resize(y, (336,320), interpolation = cv2.INTER_NEAREST)
            X = np.expand_dims(X, 0)/255
            y = np.expand_dims(y, 0)/255
            X = torch.Tensor(X)
            y = torch.Tensor(y)
            X, y = X.to(device), y.to(device)
            X = X.type(torch.cuda.FloatTensor)
            y = y.type(torch.cuda.FloatTensor)
            segmentation_pred, recon_pred = model(X)

            if(save):
                y = torch.squeeze(y) * 255
                segmentation_pred = torch.squeeze(segmentation_pred) * 255
                recon_pred =  torch.squeeze(recon_pred) * 255
                X = torch.squeeze(X)
                loss = loss_fn(segmentation_pred, y)
                test_loss += loss.item()

                cv.imwrite(os.path.join(output_dir,'recons'), recon_pred.cpu().numpy())
                cv.imwrite(os.path.join(output_dir,'masks'), segmentation_pred.cpu().numpy())

    test_loss = test_loss / len(dataloader)
    return test_loss

def test(model: torch.nn.Module,
         img_input_dir,
         mask_input_dir,
         output_dir,
         loss_fn: torch.nn.Module,
         device: torch.device,
         save: bool,
         data: str) -> Dict[str, List]:
    
    
    results = {
        "test_loss": 0
    }


    test_loss = test_step(model=model,
        img_input_dir=img_input_dir,
        mask_input_dir=mask_input_dir,
        output_dir=output_dir,
        loss_fn=loss_fn,
        device=device,
        save=save,
        data=data)
    print(
        f"test_loss: {test_loss:.4f}"
    )

    results["test_loss"] = test_loss

    return results

