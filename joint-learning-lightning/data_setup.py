import os
import glob
from re import I
import cv2 as cv
import numpy as np

import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils import data

import monai

NUM_WORKERS = os.cpu_count()



class CustomDatasetClass(data.Dataset):
    def __init__(self, dataset_path, img_size,transforms=None):
        super(CustomDatasetClass, self).__init__()
        self.img_files = glob.glob(os.path.join(dataset_path, "images", "*"))
        self.masks =  []
        self.img_size = img_size
        self.is_transform=True if transforms else False
        self.transforms=[]  
        
        if(self.is_transform):
            self.append_transforms(transforms=transforms)

        for img in self.img_files:
            self.masks.append(os.path.join(dataset_path, "ground_truth", os.path.basename(img)))
        
        
    def append_transforms(self,transforms):
        for transform in transforms:
            name=transform["name"]
            if name=="crop":
                spatial_size=transform["spatial_size"] if transform["spatial_size"] else tuple((320,336))
                self.transforms.append(monai.transforms.ResizeWithPadOrCrop(spatial_size=spatial_size))

            # elif name=="random_gaussian":
            #     prob=transform["prob"] if transform["prob"] else 0.5
            #     sigma_x=transform["sigma_x"] if transform["sigma_x"] else (1,2)
            #     self.transforms.append(monai.transforms.RandGaussianSmooth(prob=prob, sigma_x=sigma_x))
            
            elif name=="horizontal_flip":
                self.transforms.append(monai.transforms.RandFlip(prob=1.0, spatial_axis=1))
            
            elif name=="random_rotate":
                prob=transform["prob"] if transform["prob"] else 0.5
                range_x=transform["range_x"] if transform["range_x"] else [0.4,0.4]
                self.transforms.append(monai.transforms.RandRotate(prob=prob, range_x=range_x))


    def __getitem__(self, index):
        img = cv.cvtColor(cv.imread(self.img_files[index]), cv.COLOR_BGR2GRAY)
        mask = np.zeros(img.shape, dtype='uint8')
        if(os.path.exists(self.masks[index])):
            mask = cv.cvtColor(cv.imread(self.masks[index]), cv.COLOR_BGR2GRAY)

        if self.is_transform:
            composed_transform=monai.transforms.Compose(transforms=transforms)
            img=torch.Tensor(img)
            # img = img.unsqueeze(0)
            transform_input  = torch.zeros(2, 542, 562)
            transform_input[0, :, :] = img
            transform_input[1, :, :] = mask
            transformed_img= composed_transform(transform_input)
            img= torch.Tensor(transformed_img).numpy()[0, :, :]
            mask = torch.Tensor(transformed_img).numpy()[1, :, :]
        
        if(img.shape!=(320,336)):
            img = cv.resize(img, (self.img_size[1], self.img_size[0]), interpolation = cv.INTER_NEAREST)
            mask = cv.resize(mask, (self.img_size[1], self.img_size[0]), interpolation = cv.INTER_NEAREST)
        
        img = np.expand_dims(img, 0)/255
        mask = np.expand_dims(mask, 0)/255
        return img, mask

    def __len__(self):
        return len(self.img_files)

def create_dataloaders(
    dataset: str,
    batch_size: int,
    img_size: tuple,
    train_transforms:list,
    num_workers: int=40,

):
    train_dir = os.path.join(dataset, "train")
    val_dir = os.path.join(dataset, "val")
    test_dir = os.path.join(dataset, "test")

    train_data = CustomDatasetClass(train_dir, img_size,transforms=train_transforms)
    val_data = CustomDatasetClass(val_dir, img_size)
    test_data = CustomDatasetClass(test_dir, img_size)

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, val_dataloader
