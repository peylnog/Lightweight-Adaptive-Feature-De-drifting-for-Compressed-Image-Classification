'''
Date: 2023-08-21 10:39:10
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-08-30 10:28:49
FilePath: /Reproduce_PL/models.py
'''
from email import utils
from gzip import READ
from pickle import FRAME
import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
import math
from torch.nn.modules.pooling import FractionalMaxPool2d
import torchvision
from util import *
class dct(nn.Module):
    '''
    zhihu version of dct
    '''
    def __init__(self):
        super(dct, self).__init__()


        self.dct_conv = nn.Conv2d(3,192,8,8,bias=False, groups=3) # 3 h w -> 192 h/8 w/8
        self.weight = torch.from_numpy(np.load('models/DCTmtx.npy')).float().permute(2,0,1).unsqueeze(1)# 64 1 8 8, order in Z
        self.dct_conv.weight.data  =  torch.cat([self.weight] * 3, dim=0) # 192 1 8 8
        self.dct_conv.weight.requires_grad  = False

        self.mean = torch.Tensor([[[[0.485, 0.456, 0.406]]]]).reshape(1, 3, 1,
                                                                 1)
        self.std = torch.Tensor([[[[0.229, 0.224, 0.225]]]]).reshape(1, 3, 1,
                                                                1)
        self.Ycbcr = nn.Conv2d(3, 3, 1, 1, bias=False)
        trans_matrix = np.array([[0.299, 0.587, 0.114],
                                 [-0.169, -0.331, 0.5],
                                 [0.5, -0.419, -0.081]])
        trans_matrix = torch.from_numpy(trans_matrix).float().unsqueeze(
            2).unsqueeze(3)
        self.Ycbcr.weight.data = trans_matrix
        self.Ycbcr.weight.requires_grad = False

        self.reYcbcr = nn.Conv2d(3, 3, 1, 1, bias=False)
        re_matrix = np.linalg.pinv(np.array([[0.299, 0.587, 0.114],
                                 [-0.169, -0.331, 0.5],
                                 [0.5, -0.419, -0.081]]))
        re_matrix = torch.from_numpy(re_matrix).float().unsqueeze(
            2).unsqueeze(3)
        self.reYcbcr.weight.data = re_matrix

    def forward(self, x):

        # jpg = (jpg * self.std) + self.mean # 0-1
        ycbcr = self.Ycbcr(x) # b 3 h w

        dct = self.dct_conv(ycbcr)
        return dct

    def reverse(self,x):
        dct = F.conv_transpose2d(x, torch.cat([self.weight] * 3,0), bias=None, stride=8, groups = 3)
        rgb = self.reYcbcr(dct)
        return rgb


class RepConv(nn.Module):
    def __init__(self, in_channels, out_channels,  kernel_size=3, padding=1):
        super(RepConv, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0),
                                   nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
                                   nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0))
        self.conv1_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)

    def forward(self, x):

        out = self.conv1(x) + self.conv3(x) + self.conv1_1(x)
        return out

def conv3x3(in_channal,out_channal):
    return nn.Conv2d(in_channal,out_channal,3,1,1)


Conv_type = RepConv
class UNet_Dropout(nn.Module):
    def __init__(self, in_channal = 64,channal=16):
        super(UNet_Dropout, self).__init__()
        
        self.n_channels = in_channal


        self.inc = Conv_type(in_channal,channal)

        self.down1 = Conv_type(channal,channal*2)
        self.down2 = Conv_type(channal*2,channal*4)


        self.up1 = Conv_type(channal*4,channal*2)
        self.up2 = Conv_type(channal*4,channal)

        self.outc = Conv_type(channal*2, in_channal)

        self.relu = nn.PReLU()

        self.maxpool = nn.MaxPool2d(stride=2,kernel_size=2)
        self.uppool = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dropout = nn.Dropout2d(p=0.5)
    def forward(self, x):

        x1 = self.relu( self.inc(x) ) #112 
        d1 = self.maxpool(self.relu( self.down1(x1) ))  #56 
        d2 = self.maxpool(self.relu( self.down2(d1) ))   #28 

        x = self.uppool(self.relu( self.up1(d2) )) #56
        x = self.uppool(self.relu( self.up2(torch.cat([x,d1],dim=1) ) )) #112
        x = self.dropout(x)
        x = self.outc(torch.cat([x,x1],dim=1))

        return x


class UNet(nn.Module):
    def __init__(self, in_channal = 64,channal=16,out_cha=64):
        super(UNet, self).__init__()
        
        self.n_channels = in_channal


        self.inc = Conv_type(in_channal,channal)

        self.down1 = Conv_type(channal,channal*2)
        self.down2 = Conv_type(channal*2,channal*4)


        self.up1 = Conv_type(channal*4,channal*2)
        self.up2 = Conv_type(channal*4,channal)

        self.outc = Conv_type(channal*2, out_cha)

        self.relu = nn.PReLU()

        self.maxpool = nn.MaxPool2d(stride=2,kernel_size=2)
        self.uppool = nn.UpsamplingBilinear2d(scale_factor=2)
    def forward(self, x):

        x1 = self.relu( self.inc(x) ) #112 
        d1 = self.maxpool(self.relu( self.down1(x1) ))  #56 
        d2 = self.maxpool(self.relu( self.down2(d1) ))   #28 

        x = self.uppool(self.relu( self.up1(d2) )) #56
        x = self.uppool(self.relu( self.up2(torch.cat([x,d1],dim=1) ) )) #112
        x = self.outc(torch.cat([x,x1],dim=1))

        return x




class DCNN(nn.Module):
    def __init__(self, in_channal = 64,channal=32):
        super(DCNN, self).__init__()
        
        self.n_channels = in_channal
        self.c1 = nn.Conv2d(in_channal,in_channal,3,1,1)
        self.c2 = nn.Conv2d(in_channal,in_channal,3,1,1)
        self.c3 = nn.Conv2d(in_channal,in_channal,3,1,1)
        self.c4 = nn.Conv2d(in_channal,in_channal,3,1,1)

        self.relu = nn.PReLU()

    def forward(self, x):

        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        return x



class FDM_dct_rep(nn.Module):
    '''
    参数量 119800
    '''
    def __init__(self ):
        super(FDM_dct_rep, self).__init__()

        self.fdm = UNet(in_channal=64, channal=64,out_cha=64)

        self.reverse = InverShift()
        self.rgb2ycbcr = YcbcrShift()

        self.dct_conv = nn.Conv2d(1,64,8,8,bias=False)
        self.dct_conv.weight.data = torch.from_numpy(np.load('/home/date/Trans/TMM_JPEG/Reproduce_PL/DCTmtx.npy')).float().permute(2,0,1).unsqueeze(1)
        self.dct_conv.weight.requires_grad  = False

        self.guide = nn.Sequential(
            nn.Conv2d(192, 128, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 64, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, jpg):
        '''

        :param x:  feature, 64*112*112, recpetive field is 7*7
        :param jpg:  normalazitoed, 3*224*224
        :return:
        '''
        rejpg = self.reverse(jpg)
        b,c,h,w = rejpg.shape
        ycbcr = self.rgb2ycbcr(rejpg)  # b 3 h w

        dct_y = self.dct_conv(ycbcr[:, 0:1, :, :])
        dct_cb = self.dct_conv(ycbcr[:, 1:2, :, :])
        dct_cr = self.dct_conv(ycbcr[:, 2:3, :, :])
        dct = torch.cat([dct_y, dct_cb, dct_cr], dim=1)

        guide = self.guide(dct)
        fd = self.fdm(x)

        out = F.relu( fd*guide + x )

        return out





class FDM_dct_rep_HIEF(nn.Module):
    '''
    参数量 119800
    '''
    def __init__(self,input_ch ):
        super(FDM_dct_rep_HIEF, self).__init__()

        self.fdm = UNet(in_channal=input_ch, channal=input_ch//2)

        self.reverse = InverShift()
        self.rgb2ycbcr = YcbcrShift()

        self.dct_conv = nn.Conv2d(1,64,8,8,bias=False)
        self.dct_conv.weight.data = torch.from_numpy(np.load('/home/date/Trans/TMM_JPEG/Reproduce_PL/DCTmtx.npy')).float().permute(2,0,1).unsqueeze(1)
        self.dct_conv.weight.requires_grad  = False

        self.guide = nn.Sequential(
            nn.Conv2d(192, input_ch, 1, 1),
            nn.ReLU(),
            nn.Conv2d(input_ch, input_ch, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_ch, input_ch, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, jpg):
        '''

        :param x:  feature, 64*112*112, recpetive field is 7*7
        :param jpg:  normalazitoed, 3*224*224
        :return:
        '''
        rejpg = self.reverse(jpg)
        b,c,h,w = rejpg.shape
        ycbcr = self.rgb2ycbcr(rejpg)  # b 3 h w

        dct_y = self.dct_conv(ycbcr[:, 0:1, :, :])
        dct_cb = self.dct_conv(ycbcr[:, 1:2, :, :])
        dct_cr = self.dct_conv(ycbcr[:, 2:3, :, :])
        dct = torch.cat([dct_y, dct_cb, dct_cr], dim=1)


        guide = self.guide(dct)
        fd = self.fdm(x)

        out = F.relu( fd*guide + x )

        return out

class FDM_DDP(nn.Module):
    #参数量 117472
    def __init__(self):
        super(FDM_DDP, self).__init__()

        self.g1_1 = conv_relu(67, 64, 3, 1, 1) #这里表示concat了原始图像
        self.g1_2 = conv_relu(64, 64, 3, 1, 1)
        self.g2_1 = conv_relu(64, 32, 3, 1, 1)
        self.g2_2 = conv_relu(32, 32, 3, 1, 1)
        self.g3_1 = conv_relu(32, 16, 3, 1, 1)
        self.g3_2 = conv_relu(16, 16, 3, 1, 1)
        # self.w = conv_relu(224, 128, 1, 1, 0)
        self.w = nn.Conv2d(112, 64, 1, 1, 0)


    def forward(self, x,jpegs):
        jpegs = F.interpolate(jpegs,x.size()[2:])
        x1 = self.g1_1(torch.cat([x,jpegs], dim=1))
        x1 = self.g1_2(x1)

        x2 = self.g2_1(x1)
        x2 = self.g2_2(x2)

        x3 = self.g3_1(x2)
        x3 = self.g3_2(x3)

        out = F.relu(self.w(torch.cat([x1,x2,x3], dim=1)))

        return out


if __name__ == '__main__':
    jpegs = torch.randn(size=(1,3,224,224))
    feature = torch.randn(size=(1,64,112,112))
    net = FDM_dct_rep()
    
    total = sum(p.numel() for p in net.parameters())
    print(total)
    print(net(feature,jpegs).shape)
