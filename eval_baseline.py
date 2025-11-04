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
import imageio
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from unet import UNet, UNet_atten, Unet_cross_scale_transformer,Unet_cross_scale_transformer_nospatial,Unet_cross_scale_transformer_noself
from othermodels import UNet_2Plus, UNet_3Plus, SwinUnet, UNetFormer, AttentionUNet, TransUNet

channel = 1
dataname = sys.argv[2]
netname = sys.argv[3]
PATH = 'ckptsbase/' + netname + '/checkpoint_epoch' + sys.argv[1].replace("\r", "") + '.pth'
print(PATH)
if netname == 'UNet':
    model = UNet(n_channels=1, n_classes=4, bilinear=False)
elif netname == 'UNet3Plus':
    model = UNet_3Plus(n_channels=1, n_classes=4)
elif netname == 'UNet2Plus':
    model = UNet_2Plus(n_channels=1, n_classes=4)
elif netname =='UNetFormer':
    model = UNetFormer(n_classes=4)
elif netname =='SwinUNet':
    model = SwinUnet(n_channels=1, n_classes=4)
elif netname == 'AttentionUNet':
    model = AttentionUNet(img_ch=1, n_classes=4)
elif netname == 'TransUNet':
    model = TransUNet(img_dim = 512, n_classes = 4)
elif netname == 'UNet_atten':
    model = UNet_atten(n_channels=1, n_classes=4, bilinear=False)
elif netname == 'UNet_trans':
    model = Unet_cross_scale_transformer(n_channels=1, n_classes=4, bilinear=False)
elif netname =='UNet_transformer_nospatial':
    model = Unet_cross_scale_transformer_nospatial(n_channels=1, n_classes=4, bilinear=False)
elif netname =='UNet_transformer_noself':
    model = Unet_cross_scale_transformer_noself(n_channels=1, n_classes=4, bilinear=False)

model.load_state_dict(torch.load(PATH, map_location=device))
model.to(device=device)
model.eval()

def get_info(filenames, ext, root):
    images = []
    for filename in filenames :
        filepath = os.path.join(root,filename)
        if ext == '.npy':
            image = np.load(filepath)
        elif ext == '.JPG' or ext == '.tif' or ext =='.png':
            image = imageio.imread(filepath)
            w,h = image.shape
            image = cv2.resize(image, (int(h*0.25), int(w*0.25)))
        images.append(image)
    return images

def get_data(directory,ext):
    from os import listdir
    from os.path import isfile, join
    
    root_path = ""
    filenames = [f for f in listdir(directory) if isfile(join(directory, f)) and f != '.DS_Store']
    filenames = sorted(filenames)
    return filenames, get_info(filenames, ext, directory)

import argparse
import logging
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from utils.data_loading_1channel import BasicDataset as BasicDataset1channel
# from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=0.25,
                out_threshold=0.5):
    net.eval()
    if channel ==1:
        img = torch.from_numpy(BasicDataset1channel.preprocess(None, full_img, scale_factor, is_mask=False))
    elif channel ==3:
        img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()

import glob 
cwd = os.getcwd()
in_files = glob.glob('res_baseline/data/' + dataname + '/images/*.png')
for i in range(len(in_files)):
    filename = in_files[i]
    img = Image.open(filename)
    mask = predict_img(net=model,
                       full_img=img,
                       scale_factor=0.25,
                       out_threshold=0.5,
                       device=device)
    save_path = 'res_baseline/pred/' + dataname + '/' + netname
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imwrite(save_path + '/' + os.path.basename(filename), mask/3*255)
    
def iou_score(gt,pred):
    intersection = np.logical_and(gt, pred)
    union = np.logical_or(gt, pred)
    iou = np.sum(intersection) / np.sum(union)
    return iou



from eval_seg_metrics import pixel_accuracy,dice_coef,mean_accuracy

fat_dice = []
kibble_dice = []
hole_dice = []
fat_acc = []
back_dice=[]
back_acc = []
kibble_acc = []
hole_acc= []

files = sorted(os.listdir('res_baseline/pred/' + dataname + '/' + netname))
for i in range(len(files)):
    imgA = cv2.imread('res_baseline/pred/' + dataname + '/' + netname + '/' + files[i]) 
    imgB = cv2.imread('res_baseline/data/' + dataname + '/labels/' + files[i], cv2.IMREAD_GRAYSCALE)  
    imgC = cv2.imread('res_baseline/data/' + dataname + '/images/' + files[i])

    imgA = imgA[:,:,0]
    imgA = np.squeeze(imgA)
    imgA = imgA /255*3
    if len(imgB.shape)>2:
        imgB = imgB[:,:,0]
    imgB = np.squeeze(imgB)
    imgB = imgB / np.max(imgB) *3
    imgB = np.rint(imgB)
    
    imgC = imgC[:,:,0]
    imgC = np.squeeze(imgC)

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
    
    kibble_dice_score = dice_coef(kibble_imgA, kibble_imgB)
    kibble_dice.append(kibble_dice_score)
        
    hole_dice_score = dice_coef(hole_imgA, hole_imgB)
    hole_dice.append(hole_dice_score)
    
    back_dice_score = dice_coef(back_imgA, back_imgB)
    back_dice.append(back_dice_score)
    

    fat_pixel_accuracy = iou_score(fat_imgA, fat_imgB)
    if not np.isnan(fat_pixel_accuracy) and fat_pixel_accuracy!=0:
        fat_acc.append(fat_pixel_accuracy)
    
    kibble_pixel_accuracy = iou_score(kibble_imgA, kibble_imgB)
    kibble_acc.append(kibble_pixel_accuracy)

    hole_pixel_accuracy = iou_score(hole_imgA, hole_imgB)
    hole_acc.append(hole_pixel_accuracy)
    
    back_pixel_accuracy = iou_score(back_imgA, back_imgB)
    back_acc.append(back_pixel_accuracy)
    
    # fat_value = hausdorff_distance_scaled(fat_imgA, fat_imgB, scale_dict[files[i]][0], scale_dict[files[i]][1])
    # kib_value = hausdorff_distance_scaled(kibble_imgA, kibble_imgB, scale_dict[files[i]][0], scale_dict[files[i]][1])
    # hole_value = hausdorff_distance_scaled(hole_imgA, hole_imgB, scale_dict[files[i]][0], scale_dict[files[i]][1])
    # if fat_value != 0 and fat_value != np.inf:
    #     fat_eu.append(fat_value)
    # if kib_value != 0 and kib_value != np.inf:
    #     kib_eu.append(kib_value) 
    # if hole_value != 0 and hole_value != np.inf:
    #     hole_eu.append(hole_value)

a = np.mean(fat_acc)
b = np.mean(hole_acc)
c = np.mean(kibble_acc)
d = np.mean(back_acc)

e = np.mean(fat_dice)
f = np.mean(hole_dice)
g = np.mean(kibble_dice)
h = np.mean(back_dice)

# scale_dict = dict()    
# with open(dataname + "_" + netname + '.txt', 'r') as file:
#     for line in file:
#         info = line.strip().split(" ")
#         if dataname[0:2] == "AD":
#             scale_dict[info[0]] = (float(info[1]), float(info[2]))
#         else:
#             scale_dict[info[0]] = (1, 1)
file_name = "res_baseline/metric/" + dataname + "_" + netname + ".txt"
with open(file_name, 'a') as file:
    # Write new content
    file.write("checkpoint " + sys.argv[1].replace("\r", "") + "\n")
    file.write("fat_acc: " + str(round(np.mean(fat_acc), 3)) + "\n")
    file.write("hole_acc" +  str(round(np.mean(hole_acc), 3)) + "\n")
    file.write("kibble_acc" +  str(round(np.mean(kibble_acc), 3)) + "\n")
    file.write("back_acc" + str(round(np.mean(back_acc), 3)) + "\n")
    file.write("fat_dice" + str(round(np.mean(fat_dice), 3)) + "\n")
    file.write("hole_dice" +  str(round(np.mean(hole_dice), 3)) + "\n")
    file.write("kibble_dice" + str(round(np.mean(kibble_dice), 3)) + "\n")
    file.write("back_dice" + str(round(np.mean(back_dice), 3)) + "\n")
    # file.write("fat_eu" + str(round(np.mean(fat_eu) * pixel_size, 3)) + "\n")
    # file.write("kib_eu" + str(round(np.mean(kib_eu) * pixel_size, 3)) + "\n")
    # file.write("hole_eu" + str(round(np.mean(hole_eu) * pixel_size, 3)) + "\n")
    file.write('//')
    file.write("all_acc" + str(round((a+b+c)/3, 3)) + "\n")
    file.write("all_dice" + str(round((e+f+g)/3, 3)) + "\n")
    # file.write("all_eu" + str(round((np.mean(fat_eu)+np.mean(hole_eu)+np.mean(kib_eu))/3 * pixel_size, 3)) + "\n")