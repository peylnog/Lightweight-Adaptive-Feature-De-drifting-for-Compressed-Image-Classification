'''
Date: 2023-08-21 10:39:01
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-09-02 09:34:31
FilePath: /Reproduce_PL/test.py
'''
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
from models import FDM_dct_rep as Model
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


resnet = torchvision.models.resnet50().eval().cuda()
resnet = nn.DataParallel(resnet)
resnet.load_state_dict(torch.load('./ckp/checkpoint.pth.tar')['state_dict'])
resnet.eval().cuda()
resnet=resnet.module
@torch.no_grad()
def get_feature_resnet50(img):
    x = resnet.conv1(img)
    x = resnet.bn1(x)
    x = resnet.relu(x)
    return x  #torch.Size([1, 64, 112, 122])


@torch.no_grad()
def inf_resnet(img):
    x = resnet.conv1(img)
    x = resnet.bn1(x)
    x = resnet.relu(x)
    x = resnet.maxpool(x)

    x = resnet.layer1(x)
    x = resnet.layer2(x)
    x = resnet.layer3(x)
    x = resnet.layer4(x)

    x = resnet.avgpool(x)
    x = torch.flatten(x, 1)
    x = resnet.fc(x)

    return x 

@torch.no_grad()
def inf_resnet_FDM(img,model):

    x = model(get_feature_resnet50(img),img)


    x = resnet.maxpool(x)

    x = resnet.layer1(x)
    x = resnet.layer2(x)
    x = resnet.layer3(x)
    x = resnet.layer4(x)

    x = resnet.avgpool(x)
    x = torch.flatten(x, 1)
    x = resnet.fc(x)

    return x 


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
            outputs = inf_resnet(inputs)

            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)
            # if i % 60 == 0:
            #     print('itration : {}/{}   top 1: {}   top 5: {}'.format(i,len(val_loader), top1.show(), top5.show()))
            # i += 1


    return top1.avg, top5.avg
    

def val(root):
    print("Val")
    model = Model().cuda()
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


