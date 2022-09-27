import os
import cv2
import random
import numpy as np
import torch
import PIL
import torchvision

def create_augmentations(input_dir, mask_dir,  crop_size, transforms=[]):
    input_imgs = os.listdir(input_dir)
    for img_path in input_imgs:
        img = cv2.imread(os.path.join(input_dir, img_path), 0)
        mask = cv2.imread(os.path.join(mask_dir, img_path), 0)
        for t in transforms:
            if(t=='flip'):
                img_t = cv2.flip(img, 0)
                mask_t = cv2.flip(mask, 0)
                if(not os.path.exists("./flip/")):
                    os.makedirs("./flip/images/")
                    os.makedirs("./flip/masks/")
                cv2.imwrite(f'./flip/images/{img_path[:-4]}_f.jpg', img_t)
                cv2.imwrite(f'./flip/masks/{img_path[:-4]}_f.jpg', mask_t)
            elif(t=='rotate'):
                angle = random.randrange(-10,11)
                h,w = img.shape
                center = (w/2, h/2)
                rot_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
                img_t = cv2.warpAffine(src=img, M=rot_matrix, dsize=(w,h))
                mask_t = cv2.warpAffine(src=mask, M=rot_matrix, dsize=(w,h))
                if(not os.path.exists("./rotate/")):
                    os.makedirs("./rotate/images/")
                    os.makedirs("./rotate/masks/")
                cv2.imwrite(f'./rotate/images/{img_path[:-4]}_r.jpg', img_t)
                cv2.imwrite(f'./rotate/masks/{img_path[:-4]}_r.jpg', mask_t)
            elif(t=='crop'):
                h,w = img.shape
                #img_t = img_t[h/2 - crop_size[0]/2: h/2 + crop_size[0]/2][w/2 - crop_size[1]/2 : w/2 + crop_size[1]/2]
                #mask_t = mask_t[h/2 - crop_size[0]/2: h/2 + crop_size[0]/2][w/2 - crop_size[1]/2 : w/2 + crop_size[1]/2]
                img_t = img[int(h/2 - crop_size[0]/2): int(h/2 + crop_size[0]/2)][int(w/2 - crop_size[1]/2) : int(w/2 + crop_size[1]/2)]
                mask_t = mask[int(h/2 - crop_size[0]/2): int(h/2 + crop_size[0]/2)][int(w/2 - crop_size[1]/2) : int(w/2 + crop_size[1]/2)]
                if(not os.path.exists("./crop/")):
                    os.makedirs("./crop/images/")
                    os.makedirs("./crop/masks/")
                cv2.imwrite(f'./crop/images/{img_path[:-4]}_c.jpg', img_t)
                cv2.imwrite(f'./crop/masks/{img_path[:-4]}_c.jpg', mask_t)
            elif(t=='inverse_gaussian'):
                img_t = torchvision.transforms.functional.adjust_gamma(PIL.Image.fromarray(img),gamma=2,gain=1)
                img_t = np.array(img_t)
                if(not os.path.exists("./inverse_gaussian/")):
                    os.makedirs("./inverse_gaussian/images/")
                    os.makedirs("./inverse_gaussian/masks/")
                cv2.imwrite(f'./inverse_gaussian/images/{img_path[:-4]}_ig.jpg', img_t)
                cv2.imwrite(f'./inverse_gaussian/masks/{img_path[:-4]}_ig.jpg', mask)

            elif(t=='random_crop'):
                h,w = img.shape
                print(h, w)
                center_x = torch.randint(low=(crop_size[0]//2) + 1, high=h-(crop_size[0]//2) - 1, size=(1,)).item()
                center_y = torch.randint(low=(crop_size[1]//2) + 1, high=w-(crop_size[1]//2) - 1, size=(1,)).item()
                img_t = img[int(center_x - crop_size[0]/2): int(center_x + crop_size[0]/2)][int(center_y - crop_size[1]/2) : int(center_y + crop_size[1]/2)]
                mask_t = mask[int(center_x - crop_size[0]/2): int(center_x + crop_size[0]/2)][int(center_y - crop_size[1]/2) : int(center_y + crop_size[1]/2)]
                if(not os.path.exists("./random_crop/")):
                    os.makedirs("./random_crop/images/")
                    os.makedirs("./random_crop/masks/")
                cv2.imwrite(f'./random_crop/images/{img_path[:-4]}_rc.jpg', img_t)
                cv2.imwrite(f'./random_crop/masks/{img_path[:-4]}_rc.jpg', mask_t)

create_augmentations('MallesNet/train/images', 'MallesNet/train/ground_truth', (320, 336), ["inverse_gaussian"])
