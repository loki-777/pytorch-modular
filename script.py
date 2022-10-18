import os
import glob
import shutil

DIR = "checkpoints/"
TARGET = "temp/"

os.makedirs(TARGET, exist_ok=True)

for f in os.listdir(DIR):
    if int(f[6:-5]) % 20 == 19:
        shutil.move(DIR + f, TARGET + f)
