import pandas as pd
import os
import shutil
import numpy as np
import glob

df = pd.read_csv("nerve_seg_folds.csv")
DATASET = "data/NerveDataset/"

fold = 4

TARGET = f"data/fold-{fold}/"
os.makedirs(TARGET, exist_ok=True)

train = "fold_" + str(fold) + "_train"
val = "fold_" + str(fold) + "_val"
test = "fold_" + str(fold) + "_test"

train_videos = df[train].to_numpy().astype(int).astype(str)
val_videos = df[val].to_numpy().astype(int).astype(str)
test_videos = df[test].to_numpy().astype(int).astype(str)

for vid in train_videos:
    file_names = glob.glob(DATASET + "images/" + vid + "*")
    os.makedirs(TARGET+"train/images", exist_ok=True)
    for file_name in file_names:
        shutil.move(file_name, TARGET + "train/" + "images/" + os.path.basename(file_name))

    file_names = glob.glob(DATASET + "ground_truth/" + vid + "*")
    os.makedirs(TARGET+"train/ground_truth", exist_ok=True)
    for file_name in file_names:
        shutil.move(file_name, TARGET + "train/" + "ground_truth/" + os.path.basename(file_name))

for vid in val_videos:
    file_names = glob.glob(DATASET + "images/" + vid + "*")
    os.makedirs(TARGET+"val/images", exist_ok=True)
    for file_name in file_names:
        shutil.move(file_name, TARGET + "val/" + "images/" + os.path.basename(file_name))
        
    file_names = glob.glob(DATASET + "ground_truth/" + vid + "*")
    os.makedirs(TARGET+"val/ground_truth", exist_ok=True)
    for file_name in file_names:
        shutil.move(file_name, TARGET + "val/" + "ground_truth/" + os.path.basename(file_name))

for vid in test_videos:
    file_names = glob.glob(DATASET + "images/" + vid + "*")
    os.makedirs(TARGET+"test/images", exist_ok=True)
    for file_name in file_names:
        shutil.move(file_name, TARGET + "test/" + "images/" + os.path.basename(file_name))
        
    file_names = glob.glob(DATASET + "ground_truth/" + vid + "*")
    os.makedirs(TARGET+"test/ground_truth", exist_ok=True)
    for file_name in file_names:
        shutil.move(file_name, TARGET + "test/" + "ground_truth/" + os.path.basename(file_name))
