#!/usr/bin/env python
# coding: utf-8

# # *Function:*  find_concavity()
# ### This funtion is used to reduce the computational comlexity of the model 
# 

import cv2, os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from skimage import measure
from PIL import Image

def find_concavity(path,file_name,BW,ellipse_focal_ratio = 0.35, min_convexity = 0.95):

    _, mask = cv2.threshold(BW, 200, 255, cv2.THRESH_BINARY)

    num, labels = cv2.connectedComponents(mask)

    cords = []
    output = []
    SE = gen_ellipse(200,ellipse_focal_ratio)
    for index in range(num-1):
        contour = (labels==index+1).astype(np.uint8)
        x,y,w,h = cv2.boundingRect(contour)
        if measure.regionprops(contour[y:y+h,x:x+w])[0].solidity < min_convexity:
            temp = find_concavity_single_contour(contour[y:y+h,x:x+w], SE,cords = (x,y))
            if temp is not None:
                output.append(temp)
    np.save(os.path.join(path,'concavity_'+file_name[:-4]+'.npy'),output)
    return output

def find_concavity_single_contour(mask,SE,cords):

    offx, offy = cords
    concavity = []
    lock = False
    area_thresh = 25 #pixels
    dim = 25
    step = 10
    comp = 3
    while(1):
        for rot in (0,45,90,135):
            tophat = cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, get_ellipse(SE,dim,rot))
#             plt.imshow(tophat)
#             plt.show()
            if np.sum( cv2.bitwise_and(mask, cv2.bitwise_not(tophat)) ) == 0:
                lock = True
                break
            else:
                tophat = bwareaopen(tophat,area_thresh)
                num, lbl = cv2.connectedComponents(tophat)
                for index in range(num-1):
                    index += 1
                    flag, _ = cv2.connectedComponents(cv2.bitwise_and(mask,cv2.bitwise_not((lbl==index).astype(np.uint8) )))
                    if flag == comp:
#                         M = cv2.moments((lbl==index).astype(np.uint8))
#                         concavity.append( (x + int(M["m10"] / M["m00"]),y + int(M["m01"] / M["m00"])) )
                        x,y,w,h = cv2.boundingRect((lbl==index).astype(np.uint8))
                        concavity.append((x+offx,y+offy,w,h))
                        mask = cv2.bitwise_and( mask,cv2.bitwise_not((lbl==index).astype(np.uint8) ))
                        comp += 1
                        break
        dim += step

        if lock:
            break  
    if concavity == []:
        return None
    else:
        return concavity   

def gen_ellipse(size,focal_ratio):

    output = np.zeros((size,size))
    MIN = round(size/2-(size*focal_ratio/2))
    MAX = MIN + round(size*focal_ratio)
    output[MIN:MAX,:] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,round(size*focal_ratio)))
    return output

def get_ellipse(SE,size,rot):

    output = Image.fromarray(SE).rotate(rot)
    return cv2.resize(np.array(output).astype(np.uint8),(size,size))

def bwareaopen(mask,size):

    contours, _ = cv2.findContours(mask , mode = cv2.RETR_TREE,method = cv2.CHAIN_APPROX_SIMPLE)
    output = np.zeros(mask.shape[:2],dtype=np.uint8)
    for cnt in contours:
        if cv2.contourArea(cnt) > size:
            cv2.drawContours(output,[cnt], 0, (255), -1)
    return output

def annotate_concavity(img,concavity):

    for cav in concavity:
        for cord in cav:
            img[cord[1]:cord[1]+cord[3],cord[0]:cord[0]+cord[2],1] = 255
    return img   

def extract_patches(img,concavity,wr_path,file_name):
    counter = 0
    for cav in concavity:
        for cord in cav:
            name = file_name+'_'+str(counter)+'.jpg'
            out_img = cv2.cvtColor(img[cord[1]:cord[1]+cord[3],cord[0]:cord[0]+cord[2],:], cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(wr_path,name), out_img)
            counter += 1
            
