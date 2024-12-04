
import argparse
import os
import numpy as np
import math
import itertools
import time
from datetime import timedelta
import sys
import os
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import logging
import random
import torchvision
import torch
from datasets import ImageDataset_DDP
from util import AverageMeter,accuracy
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
import torch.optim as optim
import multiprocessing as mp 


resnet = torchvision.models.resnet50(pretrained=True).eval().cuda()





def validate(val_loader, model, batch_size=32):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    with torch.no_grad():
        i=0
        for inputs, labels in val_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            # print(inputs.size())
            # outputs = inf_resnet_FDM(inputs,model)
            outputs = resnet(inputs)

            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)
            # if i % 60 == 0:
            #     print('itration : {}/{}   top 1: {}   top 5: {}'.format(i,len(val_loader), top1.show(), top5.show()))
            # i += 1


    return top1.avg, top5.avg
    

def val(root):
    print("Val")
    model = resnet
    model.eval()
    import torchvision.datasets as torchdatasets
    val_batch_size=256

    # degree = os.listdir(root)
    with torch.no_grad():
        for j in root:
            path = j
            print(path)
            val_datasets = torchdatasets.ImageFolder(
            path,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ]))
            val_loader = DataLoader(val_datasets,
                                    batch_size=val_batch_size,
                                    shuffle=False,
                                    pin_memory=True,
                                    num_workers=8)

            acc1, acc5 = validate(val_loader,model,batch_size=val_batch_size)
            print("degree:{} acc1:{} acc5:{}".format(j,acc1,acc5))
        


if __name__ == '__main__':


    val(['./Data/7'])
    val(['./Data/10'])
    val(['./Data/15'])
    val(['./Data/18'])
    val(['./Data/25'])

# /ILSVRC2012_img_jpg/7
# degree:/ILSVRC2012_img_jpg/7 acc1:33.12460140306123 acc5:55.494260204081634
# Val
# /ILSVRC2012_img_jpg/10
# degree:/ILSVRC2012_img_jpg/10 acc1:47.21619897959184 acc5:71.37276785714286
# Val
# /ILSVRC2012_img_jpg/15
# degree:/ILSVRC2012_img_jpg/15 acc1:57.313456632653065 acc5:80.71428571428571
# Val
# /ILSVRC2012_img_jpg/18
# degree:/ILSVRC2012_img_jpg/18 acc1:60.40497448979592 acc5:83.02853954081633
# Val
# /ILSVRC2012_img_jpg/25
# degree:/ILSVRC2012_img_jpg/25 acc1:63.721699617346935 acc5:85.49346301020408
# Val
# /ILSVRC2012_img_val
# degree:/ILSVRC2012_img_val acc1:76.01881377551021 acc5:92.83920599489795
