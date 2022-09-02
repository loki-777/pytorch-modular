import torch
import wandb
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import cv2 as cv

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               LAMBDA,
               device: torch.device):
    model.train()
    train_loss = 0
    segmentation_loss = 0
    reconstruction_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X = X.type(torch.cuda.FloatTensor)
        y = y.type(torch.cuda.FloatTensor) / 255
        segmentation_pred, recon_score, recon_pred = model(X)

        # print(y_pred.shape, y.shape)

        loss = loss_fn(segmentation_pred, y)
        loss += (LAMBDA*recon_score)
        train_loss += loss.item()
        segmentation_loss += loss_fn(segmentation_pred, y)
        reconstruction_loss += recon_score

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = train_loss / len(dataloader)
    segmentation_loss = segmentation_loss / len(dataloader)
    reconstruction_loss = reconstruction_loss / len(dataloader)
    return train_loss, segmentation_loss, reconstruction_loss

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device,
              save,
              data) -> Tuple[float, float]:
    model.eval() 
    test_loss = 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            X = X.type(torch.cuda.FloatTensor)
            y = y.type(torch.cuda.FloatTensor) / 255
            segmentation_pred, recon_score, recon_pred = model(X)
            if(save):
                y = torch.squeeze(y) * 255
                test_pred_logits = torch.squeeze(segmentation_pred) * 255
                X = torch.squeeze(X)
                img = torch.cat((X, y, test_pred_logits), 0)
                if data == "test":
                    cv.imwrite("../data/test_outputs/"+f"{batch}.jpg", img.cpu().numpy())
                else:
                    cv.imwrite("../data/val_outputs/"+f"{batch}.jpg", img.cpu().numpy())
            loss = loss_fn(segmentation_pred, y) + recon_score
            test_loss += loss.item()

    test_loss = test_loss / len(dataloader)
    return test_loss

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          val_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          scheduler,
          loss_fn: torch.nn.Module,
          epochs: int,
          LAMBDA: float,
          wandb_params: dict,
          device: torch.device) -> Dict[str, List]:
    results = {"train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(epochs):
        train_loss, segmentation_loss, reconstruction_loss = train_step(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            LAMBDA=LAMBDA,
                                            device=device)
        val_loss = test_step(model=model,
            dataloader=val_dataloader,
            loss_fn=loss_fn,
            device=device,
            save=False,
            data="val")
        scheduler.step()
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"segmentation_loss: {segmentation_loss:.4f} | "
            f"reconstruction_loss: {reconstruction_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
        )

        if(wandb_params["log"] == True):
            if "train_loss" in wandb_params["fields"]:
                wandb.log({"train_loss": train_loss})
                wandb.log({"segmentation_loss": segmentation_loss})
                wandb.log({"reconstruction_loss": reconstruction_loss})
            if "val_loss" in wandb_params["fields"]:
                wandb.log({"val_loss": val_loss})

        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)

    return results

def test(model: torch.nn.Module,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          device: torch.device,
          save: bool,
          data: str) -> Dict[str, List]:
    results = {
        "test_loss": 0
    }


    test_loss = test_step(model=model,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
        device=device,
        save=save,
        data=data)
    print(
        f"test_loss: {test_loss:.4f}"
    )

    results["test_loss"] = test_loss

    return results

