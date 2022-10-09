import torch
import torch.nn as nn
import os
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import cv2 as cv
import numpy as np
from model import JointModelLightning
from PIL import Image
import PIL
import csv
from collections import defaultdict
import sys
import pandas as pd

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5,0.6, 0.7, 0.8, 0.9]

OVERLAP_PERCENTAGES = [25,50,60,70]
fp_area_thresh = int(0.1*3663.0)
nerve = 'sc'

def dice(y_true, y_pred):
    smooth = 1.
    #print(y_true.shape, y_pred.shape)
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return score

def hypothesis_test(y_true, y_pred, iou_threshold=0.5, area_threshold=225):
    smooth = 1.
    test = None
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    pred_area = np.sum(y_pred_f)
    gt_area   = np.sum(y_true_f)
    intersection = np.sum(y_true_f * y_pred_f)
    union = gt_area + pred_area - intersection
    iou = (intersection + smooth)/(union + smooth)
    
    if pred_area < area_threshold :
        if gt_area ==  0: test = 'tn'  # Negative Data (Nothing in Ground Truth) 
        else: test = 'fn'              # Miss
    else:                              # Significant prediction
        if iou >= iou_threshold: test = 'tp'
        else: test = 'fp'
    return test

def frame_to_video_evaluations(frames, dice_scores, hypo_tests, nerve, joint = False) :
    
    if joint == True:
        nerves = ['sc' , 'isc']
    else:
        nerves = [nerve]

    vid_tests = defaultdict(list) 
    vid_dice  = defaultdict(list)     
        
    for nerve in nerves:
        for thresh in range(len(THRESHOLDS)): 
            #print(nerve, thresh, frames)
            for frame in frames: 
                frame_dice = dice_scores[frame][nerve][thresh]
                vid_dice[nerve + '_th_' + str(thresh)].append(frame_dice)                
                for pct in OVERLAP_PERCENTAGES: 
                    frame_test = hypo_tests[frame][nerve + '_pct_' +str(pct)][thresh]                    
                    vid_tests[nerve + '_pct_' +str(pct)+'_th_' + str(thresh)].append(frame_test) 

    avg_dice = defaultdict(list)    
    tests = defaultdict(list)
    precision_series = defaultdict(list)
    recall_series = defaultdict(list)
    specificity_series = defaultdict(list)
    f_score_series = defaultdict(list)  
    
    for nerve in nerves:
        for thresh in range(len(THRESHOLDS)): 
            
            threshold_avg_dice = sum(vid_dice[nerve + '_th_' + str(thresh)])/len(vid_dice[nerve + '_th_' + str(thresh)])
            avg_dice[nerve].append(threshold_avg_dice) 
            
            for pct in OVERLAP_PERCENTAGES: 
                tp = vid_tests[nerve +'_pct_'+str(pct)+'_th_' + str(thresh)].count('tp') 
                tn = vid_tests[nerve +'_pct_'+str(pct)+'_th_' + str(thresh)].count('tn')
                fp = vid_tests[nerve +'_pct_'+str(pct)+'_th_' + str(thresh)].count('fp')
                fn = vid_tests[nerve +'_pct_'+str(pct)+'_th_' + str(thresh)].count('fn') 
                
                tests[nerve+'_pct_'+str(pct)+'_tp'].append(tp)
                tests[nerve+'_pct_'+str(pct)+'_tn'].append(tn)
                tests[nerve+'_pct_'+str(pct)+'_fp'].append(fp)
                tests[nerve+'_pct_'+str(pct)+'_fn'].append(fn) 
                
                try:
                    precision = tp/(tp + fp)
                except:
                    if len(precision_series[nerve+ '_pct_'+str(pct)]) > 0 : 
                        if precision_series[nerve+ '_pct_'+str(pct)][-1] is not None:
                            if precision_series[nerve+ '_pct_'+str(pct)][-1] > 0.99:
                                precision = 1.0
                            else: precision = None
                        else: precision = None
                    else: precision = None
                precision_series[nerve+'_pct_'+str(pct)].append(precision)

                try:
                    recall = tp/(tp + fn)
                except: recall = None
                recall_series[nerve+'_pct_'+str(pct)].append(recall)

                try:
                    specificity = tn/(tn + fp)
                except: specificity = None
                specificity_series[nerve +'_pct_'+str(pct)].append(specificity)

                try:
                    f_score = 2*tp/(2*tp + fp + fn)
                except: f_score = None
                f_score_series[nerve +'_pct_'+str(pct)].append(f_score)  
                
    df = pd.DataFrame({'threshold': THRESHOLDS}) 
    for nerve in nerves:   
        df['Dice_' + nerve] = avg_dice[nerve]        
        
    for nerve in nerves:    
        for pct in OVERLAP_PERCENTAGES: 
            df[nerve+'_pct_'+str(pct)+'_tp'] = tests[nerve+'_pct_'+str(pct)+'_tp']
            df[nerve+'_pct_'+str(pct)+'_tn'] = tests[nerve+'_pct_'+str(pct)+'_tn']
            df[nerve+'_pct_'+str(pct)+'_fp'] = tests[nerve+'_pct_'+str(pct)+'_fp']
            df[nerve+'_pct_'+str(pct)+'_fn'] = tests[nerve+'_pct_'+str(pct)+'_fn']
            df[nerve+'_pct_'+str(pct)+'_precision'] = precision_series[nerve+'_pct_'+str(pct)] 
            df[nerve+'_pct_'+str(pct)+'_recall'] = recall_series[nerve+'_pct_'+str(pct)] 
            df[nerve+'_pct_'+str(pct)+'_specificity'] = specificity_series[nerve+'_pct_'+str(pct)] 
            df[nerve+'_pct_'+str(pct)+'_f_score'] = f_score_series[nerve+'_pct_'+str(pct)]  
    return df

def test_step(model: torch.nn.Module, 
              vid_input_dir,
              mask_input_dir,
              img_size,
              output_dir,
              device: torch.device,
              save) -> Tuple[float, float]:
     
    test_loss = 0

    model.to(device)
    model.eval()


    if(not os.path.exists(output_dir)):
        os.mkdir(output_dir)
        os.mkdir(os.path.join(output_dir,'reconstruction'))
        os.mkdir(os.path.join(output_dir,'masks'))
        os.mkdir(os.path.join(output_dir,'CSV'))

    vid_path_list = os.listdir(vid_input_dir)

    test_loss = 0
    overall_recon_loss = 0

    dice_scores = []
    hypo_tests = []
    dice_scores_vid = None
    hypo_scores_vid = None
    dice_scores_img = None
    hypo_scores_img = None

    with torch.inference_mode():
        
        for vid_path in vid_path_list:
            print(vid_path)
            img_path_list = os.listdir(os.path.join(vid_input_dir,vid_path))
            os.makedirs(os.path.join(output_dir,'reconstruction', os.path.basename(vid_path)), exist_ok=True)
            os.makedirs(os.path.join(output_dir,'masks', os.path.basename(vid_path)), exist_ok=True)
            # os.makedirs(os.path.join(output_dir,'CSV', os.path.basename(vid_path)), exist_ok=True)

            dice_scores_vid = defaultdict(list)
            hypo_tests_vid = defaultdict(list)

            for img_path in img_path_list:
                dice_scores_img = defaultdict(list)
                hypo_tests_img = defaultdict(list)

                X = cv.imread(os.path.join(vid_input_dir, vid_path, img_path),0)
                y = np.zeros(X.shape, dtype='uint8')
                if(os.path.exists(os.path.join(mask_input_dir, vid_path, img_path))):
                    y = cv.imread(os.path.join(mask_input_dir, vid_path, img_path),0)
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
                # segmentation_pred = (segmentation_pred > threshold) + 0 
                segmentation_pred = torch.squeeze(segmentation_pred).cpu().numpy()
                recon_pred = np.squeeze(recon_pred)
                y = torch.squeeze(y).cpu().numpy()
                y = (y > 0) + 0

                for i in range(len(THRESHOLDS)):
                    thresh  = THRESHOLDS[i]
                    pred  = (segmentation_pred  > thresh ) + 0
                    dice_score  = dice(y , pred)    
                    dice_scores.append({"video": vid_path, "image": img_path, "thresh": thresh, "dice_score": dice_score})
                    dice_scores_img[nerve].append(dice_score) 

                    for pct in OVERLAP_PERCENTAGES:
                        hypo_tests.append({"video": vid_path, "image": img_path, "thresh": thresh, "overlap_percentage": pct,
                            "hypothesis": hypothesis_test(y , segmentation_pred , pct/100, fp_area_thresh)})
                        hypo_tests_img[nerve+'_pct_' +str(pct)].append(hypothesis_test(y , segmentation_pred , pct/100, fp_area_thresh))

                dice_scores_vid[img_path] = dice_scores_img
                hypo_tests_vid[img_path] = hypo_tests_img

                if(True):
                    segmentation_pred *= 255
                    recon_pred *= 255

                    recon_pred = np.array(recon_pred, dtype='uint8').T
                    segmentation_pred = np.array(segmentation_pred, dtype='uint8')

                    recon_pred=Image.fromarray(recon_pred)
                    segmentation_pred=Image.fromarray(segmentation_pred)

                    recon_pred.save(os.path.join(output_dir,'reconstruction', os.path.basename(vid_path), img_path[:-4]+'.png'),"PNG", dpi=(300, 300))
                    segmentation_pred.save(os.path.join(output_dir,'masks', os.path.basename(vid_path), img_path[:-4]+'.png'), "PNG", dpi=(300, 300))

            vid_eval = frame_to_video_evaluations(img_path_list, dice_scores_vid, hypo_tests_vid, nerve)
            eval_csv_path = os.path.join(output_dir, 'CSV', vid_path + '.csv')
            vid_eval.to_csv(eval_csv_path, index=False)
    
    with open("dice_scores.csv", 'w',newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["video","image","thresh","dice_score"])
        writer.writeheader()
        for key in dice_scores:
            writer.writerow(key)
    
    with open("hypo_tests.csv", 'w',newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["video","image","thresh","overlap_percentage","hypothesis"])
        writer.writeheader()
        for key in hypo_tests:
            writer.writerow(key)

def test(model: torch.nn.Module,
         vid_input_dir,
         mask_input_dir,
         img_size,
         output_dir,
         device: torch.device,
         save: bool) -> Dict[str, List]:

    test_step(model=model,
        vid_input_dir=vid_input_dir,
        mask_input_dir=mask_input_dir,
        img_size=img_size,
        output_dir=output_dir,
        device=device,
        save=save)

#CONFIG_PATH = "configs/"
#config_name = sys.argv[1]

#config = load_config(CONFIG_PATH, config_name)

model = JointModelLightning.load_from_checkpoint('run5.ckpt')

# model = JointModelLightning.load_from_checkpoint(
#         in_channels=config["model_parameters"]["in_channels"],
#         img_size=(config["model_parameters"]["img_size_w"], config["model_parameters"]["img_size_h"]),
#         patch_size=config["model_parameters"]["patch_size"],
#         decoder_dim=config["model_parameters"]["decoder_dim"],
#         masking_ratio=config["model_parameters"]["masking_ratio"],
#         out_channels=config["model_parameters"]["out_channels"],
#         LAMBDA=config["training_parameters"]["lambda"],
#         NUM_EPOCHS=config["training_parameters"]["num_epochs"],
#         LEARNING_RATE=config["training_parameters"]["learning_rate"],
#         WARMUP_EPOCHS=config["training_parameters"]["warmup_epochs"],
#         WEIGHT_DECAY=config["training_parameters"]["weight_decay"]
#         )

#model.load_state_dict(torch.load("Baseline/best_val_acc.pth"), strict=False)

test(model=model, 
vid_input_dir="../data/NerveDataset/test/videos",
mask_input_dir="../data/NerveDataset/test/masks",
img_size=(320, 336),
output_dir="../outputs",
device="cuda:0",
save=True)
