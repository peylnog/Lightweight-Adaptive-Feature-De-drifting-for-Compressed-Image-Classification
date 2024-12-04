
import glob
import random
import os
import numpy as np
import torch
import cv2
import torchvision.transforms

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import time
from tqdm import tqdm
import random
from os.path import join
from pillow_heif import register_heif_opener
register_heif_opener()


class ImageDataset_DDP(Dataset):
    def __init__(self,input_root,gt_root,all=False,format='.png'):

        self.input_root=input_root
        self.gt_root=gt_root
        self.format=format
        if all == False:
            self.img_names = [os.path.join(input_root,x) for x in os.listdir(self.input_root)]
        else:
            self.img_names=[]
            f = os.listdir(self.input_root)
            for i in f:
                img_names = os.listdir(os.path.join(input_root,i))
                for names in img_names:
                    self.img_names.append( os.path.join(input_root,i,names) )

        self.tf = transforms.Compose([

            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
                                 
        ])


        self.crop_size = 224


    def __getitem__(self, index):


        img_name = self.img_names[index % len(self.img_names)]
        gt_name =  img_name.split('/')[-1].split('_')[0] +self.format
        # gt_name =  img_name.split('/')[-1].split('.')[0] +self.format

        img_input = Image.open(img_name)
        img_gt =  Image.open(os.path.join(self.gt_root,gt_name))



        i, j, h, w = transforms.RandomCrop.get_params(img_input, output_size=(self.crop_size, self.crop_size))
        img_input = TF.crop(img_input, i, j, h, w)
        img_gt = TF.crop(img_gt, i, j, h, w)


        if random.random() > 0.1:
            angle = random.randint(-180, 180)
            img_input = TF.rotate(img_input, angle)
            img_gt = TF.rotate(img_gt, angle)


        return self.tf(img_input) , self.tf(img_gt) 



    def __len__(self):
        return len(self.img_names)

