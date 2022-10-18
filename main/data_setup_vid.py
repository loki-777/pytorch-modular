import os
import glob
from re import I
import cv2 as cv
import numpy as np
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils import data
import torch
import PIL

NUM_WORKERS = os.cpu_count()

class CustomDatasetClass(data.Dataset):
    def __init__(self, dataset_path, img_size, augmentation=None):
        super(CustomDatasetClass, self).__init__()
        self.dirs = os.listdir(os.path.join(dataset_path, "images"))
        self.img_list = []
        for vid in self.dirs:
            self.img_list.append(sorted(glob.glob(os.path.join(dataset_path, "images", vid, "*"))))
        self.masks =  []
        self.img_size = img_size
        self.augmentation=augmentation
        for vid in self.img_list:
            temp = []
            for img in vid:
                temp.append(img.replace("images", "ground_truth"))
            self.masks.append(sorted(temp))


    def flip(self, img, mask):
        img_t = transforms.functional.hflip(PIL.Image.fromarray(img))
        mask_t = transforms.functional.hflip(PIL.Image.fromarray(mask))
        return np.array(img_t), np.array(mask_t)

    def random_rotate(self, img, mask):
        angle = random.randrange(-10,11)
        h,w = img.shape
        center = (w/2, h/2)
        rot_matrix = cv.getRotationMatrix2D(center=center, angle=angle, scale=1)
        img_t = cv.warpAffine(src=img, M=rot_matrix, dsize=(w,h))
        mask_t = cv.warpAffine(src=mask, M=rot_matrix, dsize=(w,h))
        return img_t, mask_t

    def center_crop(self, img, mask):
        h,w = img.shape
        crop_size = (336, 352)
        img_t = img[int(h/2 - crop_size[0]/2): int(h/2 + crop_size[0]/2)][int(w/2 - crop_size[1]/2) : int(w/2 + crop_size[1]/2)]
        mask_t = mask[int(h/2 - crop_size[0]/2): int(h/2 + crop_size[0]/2)][int(w/2 - crop_size[1]/2) : int(w/2 + crop_size[1]/2)]
        return img_t, mask_t

    def inverse_gaussian(self, img, mask):
        img_t = transforms.functional.adjust_gamma(PIL.Image.fromarray(img),gamma=2,gain=1)
        return np.array(img_t), mask

    def random_crop(self, img, mask):
        h,w = img.shape
        crop_size = (336, 352)
        center_x = torch.randint(low=(crop_size[0]//2) + 1, high=h-(crop_size[0]//2) - 1, size=(1,)).item()
        center_y = torch.randint(low=(crop_size[1]//2) + 1, high=w-(crop_size[1]//2) - 1, size=(1,)).item()
        img_t = img[int(center_x - crop_size[0]/2): int(center_x + crop_size[0]/2)][int(center_y - crop_size[1]/2) : int(center_y + crop_size[1]/2)]
        mask_t = mask[int(center_x - crop_size[0]/2): int(center_x + crop_size[0]/2)][int(center_y - crop_size[1]/2) : int(center_y + crop_size[1]/2)]
        return img_t, mask_t

    def __getitem__(self, index):

        img_3d = np.zeros((10, self.img_size[0], self.img_size[1]))
        i = 0
        for img_path in self.img_list[index]:
            img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2GRAY)
            img = cv.resize(img, (self.img_size[1], self.img_size[0]), interpolation = cv.INTER_NEAREST)
            img_3d[i, :, :] = img.copy()
            i += 1

        mask_3d = np.zeros((10, self.img_size[0], self.img_size[1]))
        i = 0
        for img_path in self.masks[index]:
            if(os.path.exists(img_path)):
                img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2GRAY)
            else:
                img = np.zeros((542, 562))
            img = cv.resize(img, (self.img_size[1], self.img_size[0]), interpolation = cv.INTER_NEAREST)
            mask_3d[i, :, :] = img.copy()
            i += 1

        #img_3d = np.expand_dims(img_3d, 0)/255
        #mask_3d = np.expand_dims(mask_3d, 0)/255

        img_3d = img_3d / 255
        mask_3d = mask_3d / 255
        return img_3d, mask_3d
        
    def __len__(self):
        return len(self.img_list)

def create_dataloaders(
    dataset: str,
    batch_size: int,
    img_size: tuple,
    num_workers: int=16,
    augmentation="transform"
):
    train_dir = os.path.join(dataset, "train")
    val_dir = os.path.join(dataset, "val")
    test_dir = os.path.join(dataset, "test")

    train_data = CustomDatasetClass(dataset, img_size, augmentation=augmentation)
    val_data = CustomDatasetClass(dataset, img_size)
    test_data = CustomDatasetClass(dataset, img_size)

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
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, val_dataloader
