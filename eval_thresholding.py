import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import os,sys
import scipy.io as scio 
import matplotlib.pyplot as plt
import math
import pprint
import cv2
import scipy.ndimage as ndimage
from eval_seg_metrics import pixel_accuracy,dice_coef,mean_accuracy
import glob
import os
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from utils.data_loading import BasicDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




def evaluateall(data):

    im_path = '../seg_data/'+data+'/images/*.png'
    gt_path = '../seg_data/'+data+'/labels/*.png'
    pred_path = './res_thresholding/'+data + '/*.png'

       

    ########### get evaluation result
    filenames_pred = sorted(glob.glob(pred_path))
    filenames_gt = sorted(glob.glob(gt_path))
    filenames_im = sorted(glob.glob(im_path))


    def iou_score(gt,pred):
        """
        Compute Intersection over Union (IoU) score between ground truth and predicted segmentation masks.
        
        Parameters:
        - gt: numpy array of the ground truth segmentation mask.
        - pred: numpy array of the predicted segmentation mask.
        
        Returns:
        - iou: The IoU score as a float.
        """
        intersection = np.logical_and(gt, pred)
        union = np.logical_or(gt, pred)
        iou = np.sum(intersection) / np.sum(union)
        return iou



    fat_dice = []
    kibble_dice = []
    hole_dice = []
    fat_acc = []
    back_dice=[]
    back_acc = []
    kibble_acc = []
    hole_acc= []


    ################read and calculate

    for i in range(len(filenames_pred)):
        
        imgA = cv2.imread(filenames_pred[i]) 
    #     print(np.unique(imgA))
        imgB = cv2.imread(filenames_gt[i], cv2.IMREAD_GRAYSCALE)  
        

    #     w,h,c = imgB.shape
    #     imgB = cv2.resize(imgB, (int(h*0.25), int(w*0.25)))
        
        imgC = cv2.imread(filenames_im[i])
    #     print(np.unique(imgC))

    #     w,h,c = imgC.shape
    #     imgC = cv2.resize(imgC, (int(h*0.25), int(w*0.25)))
        
        # print(filenames_pred[i])
        # print(filenames_gt[i])
        # print(filenames_im[i])


        imgA = imgA[:,:,0]
        imgA = np.squeeze(imgA)
        imgA = imgA /255*3
        # print(np.unique(imgA))
    #     #########post processing
    #     tmp1 = np.linspace(-1, 1, imgA.shape[1])
    #     tmp2 = np.linspace(-1, 1, imgA.shape[0])
    #     X,Y = np.meshgrid(tmp1,tmp2)
    #     Z = np.sqrt(X**2 + Y**2)
        
    #     imgA[Z>0.74]=0
    #     ###############
        
        if len(imgB.shape)>2:
            imgB = imgB[:,:,0]
        imgB = np.squeeze(imgB)
        imgB = imgB / np.max(imgB) *3
        imgB = np.rint(imgB)
        # print(np.unique(imgB))
        
        imgC = imgC[:,:,0]
        imgC = np.squeeze(imgC)
        
    #     print(imgA.shape)
    #     print(imgB.shape)
        # re = np.concatenate((imgA,imgB,imgC/255*3),1)
        # plt.imshow(re,cmap='gray')
        # plt.show()

        
        
        #kibble index=1
        kibble_imgA = imgA.copy()
        kibble_imgA[imgA==2] = 0
        kibble_imgA[imgA==3] = 0
        
        kibble_imgB = imgB.copy()
        kibble_imgB[imgB==2] = 0
        kibble_imgB[imgB==3] = 0
        
      
        
        #fat index=2
        fat_imgA = imgA.copy()
        fat_imgA[imgA==1] = 0
        fat_imgA[imgA==3] = 0
        fat_imgA[fat_imgA==2] = 1
        fat_imgB = imgB.copy()
        fat_imgB[imgB==1] = 0
        fat_imgB[imgB==3] = 0
        fat_imgB[fat_imgB==2] = 1
        

         
        #space index=3
        hole_imgA = imgA.copy()
        hole_imgA[imgA==1] = 0
        hole_imgA[imgA==2] = 0

        hole_imgA[hole_imgA==3] = 1
        hole_imgB = imgB.copy()
        hole_imgB[imgB==1] = 0
        hole_imgB[imgB==2] = 0
        hole_imgB[hole_imgB==3] = 1



        
        #kibble index=0
        back_imgA=(~imgA.astype(bool)).astype(int)
        back_imgB=(~imgB.astype(bool)).astype(int)
        
        
        
        
        fat_dice_score = dice_coef(fat_imgA, fat_imgB)
        if fat_dice_score!=1 and fat_dice_score>0.1:
            fat_dice.append(fat_dice_score)
        # print('fat dice:',fat_dice_score)
        kibble_dice_score = dice_coef(kibble_imgA, kibble_imgB)
        kibble_dice.append(kibble_dice_score)
        
            
        hole_dice_score = dice_coef(hole_imgA, hole_imgB)
        hole_dice.append(hole_dice_score)
        
        
        back_dice_score = dice_coef(back_imgA, back_imgB)
        back_dice.append(back_dice_score)
        

        fat_pixel_accuracy = iou_score(fat_imgA, fat_imgB)
        # print(fat_pixel_accuracy)
        if not np.isnan(fat_pixel_accuracy) and fat_pixel_accuracy!=0:
            fat_acc.append(fat_pixel_accuracy)
        
        kibble_pixel_accuracy = iou_score(kibble_imgA, kibble_imgB)
        kibble_acc.append(kibble_pixel_accuracy)

        hole_pixel_accuracy = iou_score(hole_imgA, hole_imgB)
        hole_acc.append(hole_pixel_accuracy)
        
        back_pixel_accuracy = iou_score(back_imgA, back_imgB)
        back_acc.append(back_pixel_accuracy)




    a = np.mean(fat_acc)
    b = np.mean(hole_acc)
    c = np.mean(kibble_acc)
    d = np.mean(back_acc)

    e = np.mean(fat_dice)
    f = np.mean(hole_dice)
    g = np.mean(kibble_dice)
    h = np.mean(back_dice)

    print('/////////////////////////////////')
    print('thresholding with ',data)
    print('/////////////////////////////////')

    print(a)
    print(b)
    print(c)
    print(d)
    print(e)
    print(f)
    print(g)
    print(h)
        # print('fat accuracy:',a)
    # print('pore accuracy:', b)
    # print('kibble accuracy:', c)
    # print('background accuracy:', d)
    # print('fat dice:', e)
    # print('pore dice:', f)
    # print('kibble dice:', g)
    # print('background dice:', h)
    print('//')
    print((a+b+c)/3)
    print((e+f+g)/3)


data =[ 'highfirst','highsec','highthird']
for i in range(len(data)):
    evaluateall(data[i])




