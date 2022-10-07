import cv2
import os
import numpy as np
from skimage.color import label2rgb
import subprocess

FPS = 8

def convert_avi2mp4(video_path):
    out_file = video_path.replace('.avi','.mp4')
    ffmpeg_cmd = 'ffmpeg -hide_banner -loglevel error -i ' + video_path + '  ' + out_file
    #print(out_file)
    subprocess.call(ffmpeg_cmd, shell=True)
    remove_cmd = 'rm ' + video_path
    subprocess.call(remove_cmd, shell=True) 

def generate_test_vid(video_path, vid_annotations, vid_pred, fps=FPS):
    
    if(not os.path.exists('masked_videos')):
        os.mkdir('masked_videos')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    height = 320
    width  = 336 
    vid_out_path  = 'masked_videos/' + video_path[-4:] + '.avi'   
    
    spatial_resolution = (width , height) 
    out = cv2.VideoWriter(vid_out_path, fourcc, fps, spatial_resolution, isColor = True)
    frame_num = 0 
    
    img_list = os.listdir(video_path)
    img_list.sort()
    
    for i in img_list:
        frame = cv2.imread(os.path.join(video_path,i))
        frame = cv2.resize(frame, (336,320))
        contoured = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gt = np.zeros(frame.shape)
        #print(os.path.join(vid_annotations,i), os.path.exists(os.path.join(vid_annotations,i)))
        if (os.path.exists(os.path.join(vid_annotations,i))):
            gt = cv2.imread(os.path.join(vid_annotations,i),0)
        
        gt = cv2.resize(gt, (336,320))
        gt = (gt > 127).astype('uint8')

        #cv2.imwrite('temp/'+i,gt)
        pred_sc = cv2.imread(os.path.join(vid_pred,i[:3]+'.png'),0)
        pred_sc = cv2.resize(pred_sc,(336,320))
        
        labels = np.zeros(frame.shape) 
        labels = pred_sc   # Blue
        #labels[np.where(gt == 1)] = 2
        #labels[np.where((gt == 1)&(pred_sc==1))] = 3
        color_pallette = [(255,0,0)]     
        
        #print(labels.shape, contoured.shape)
        overlayed = label2rgb(labels,contoured, colors = color_pallette,
                                    alpha  = 0.01, bg_label=0,bg_color=None)                                     
        overlayed[overlayed > 1.0] = 1.0  # avoid clipping warning
        overlayed = overlayed*255
        
        gt_contours = cv2.findContours(gt.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[-2] 
        for contour in gt_contours:  
            cv2.drawContours(overlayed, contour, -1, (0,   0,   255), 1) # Red               
        
        out.write(overlayed.astype('uint8'))  
        frame_num += 1
        #print (frame.shape, gt_shape)
        
    out.release() 
    #convert_avi2mp4(vid_out_path)

test_vids = os.listdir('./data/NerveDataset/test/videos')
for vid in test_vids:
    print(vid)
    generate_test_vid('./data/NerveDataset/test/videos/'+vid, './data/NerveDataset/test/masks/'+vid, './outputs/masks/'+vid, fps=FPS)    
