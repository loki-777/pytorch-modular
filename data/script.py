import os
import shutil
import random

dirs = ["random_crop/", "inverse_gaussian/", "crop/", "flip/", "rotate/"]
TARGET = "NerveDatasetAug/"
n = 1400
for d in dirs:
    images = os.listdir(d + "images/")
    random.shuffle(images)
    for img in images[:n]:
        shutil.copy(d + "images/" + img, TARGET + "images/" + img)
        shutil.copy(d + "masks/" + img, TARGET + "ground_truth/" + img)
