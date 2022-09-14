import os
import cv2
import random

def create_augmentations(input_dir, mask_dir,  crop_size, transforms=[]):
    input_imgs = os.listdir(input_dir)
    for img_path in input_imgs:
        img = cv2.imread(os.path.join(input_dir, img_path), 0)
        mask = cv2.imread(os.path.join(mask_dir, img_path), 0)
        for t in transforms:
            if(t=='flip'):
                img_t = cv2.flip(img)
                mask_t = cv2.flip(mask)
                if(not os.path.exists("./flip/")):
                    os.mkdir("./flip/images/")
                    os.mkdir("./flip/masks/")
                cv2.imwrite(f'./flip/images/{img_path[:-3]}_f.jpg', img_t)
                cv2.imwrite(f'./flip/mask/{img_path[:-3]}_f.jpg', mask_t)
            elif(t=='rotate'):
                angle = random.randrange(-10,11)
                h,w = img.shape
                center = (w/2, h/2)
                rot_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
                img_t = cv2.warpAffine(src=img, M=rot_matrix, dsize=(w,h))
                mask_t = cv2.warpAffine(src=mask, M=rot_matrix, dsize=(w,h))
                if(not os.path.exists("./rotate/")):
                    os.mkdir("./rotate/images/")
                    os.mkdir("./rotate/masks/")
                cv2.imwrite(f'./rotate/images/{img_path[:-3]}_r.jpg', img_t)
                cv2.imwrite(f'./rotate/mask/{img_path[:-3]}_r.jpg', mask_t)
            elif(t=='crop'):
                h,w = img.shape
                img_t = img_t[h/2 - crop_size[0]/2: h/2 + crop_size[0]/2][w/2 - crop_size[1]/2 : w/2 + crop_size[1]/2]
                mask_t = mask_t[h/2 - crop_size[0]/2: h/2 + crop_size[0]/2][w/2 - crop_size[1]/2 : w/2 + crop_size[1]/2]
                if(not os.path.exists("./crop/")):
                    os.mkdir("./crop/images/")
                    os.mkdir("./crop/masks/")
                cv2.imwrite(f'./crop/images/{img_path[:-3]}_c.jpg', img_t)
                cv2.imwrite(f'./crop/mask/{img_path[:-3]}_c.jpg', mask_t)

create_augmentations('NerveDataset/train/images', 'NerveDataset/train/ground_truth')
