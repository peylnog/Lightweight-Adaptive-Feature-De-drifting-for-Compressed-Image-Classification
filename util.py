# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :dejpeg
# @File     :util
# @Date     :2021/2/22 下午9:52
# @Author   :SYJ
# @Email    :JuZiSYJ@gmail.com
# @Software :PyCharm
-------------------------------------------------
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
import math
from PIL import Image
from matplotlib import pyplot as plt
from ptflops import get_model_complexity_info

def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


    def show(self):
        return self.avg


def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
    lr = lr / 5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def conv_dw(inp, oup, kernel_size, stride, pad=0, bias = True):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size, stride, pad, bias=True, groups=inp),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        # nn.ReLU(inplace=True)
    )

def conv_relu(inp, oup, kernel_size, stride, pad=0, bias = True, act='relu'):
    if act == 'relu':
        return nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=True),
            # nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
        )
    elif act == 'prelu':
        return nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=True),
            # nn.BatchNorm2d(oup),
            nn.PReLU(oup)
        )



class DWTForward(nn.Module):
    def __init__(self):
        super(DWTForward, self).__init__()
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
        filts = np.stack([ll[None, ::-1, ::-1], lh[None, ::-1, ::-1],
                          hl[None, ::-1, ::-1], hh[None, ::-1, ::-1]],
                         axis=0)
        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad=False)

    def forward(self, x):
        C = x.shape[1]
        filters = torch.cat([self.weight, ] * C, dim=0)
        y = F.conv2d(x, filters, groups=C, stride=2)
        return y


class DWTInverse(nn.Module):
    def __init__(self):
        super(DWTInverse, self).__init__()
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
        filts = np.stack([ll[None, ::-1, ::-1], lh[None, ::-1, ::-1],
                          hl[None, ::-1, ::-1], hh[None, ::-1, ::-1]],
                         axis=0)
        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad=False)

    def forward(self, x):
        C = int(x.shape[1] / 4)
        filters = torch.cat([self.weight, ] * C, dim=0)
        y = F.conv_transpose2d(x, filters, groups=C, stride=2)
        return y

class DCT(nn.Module):
    def __init__(self, N = 8, in_channal = 3):
        super(DCT, self).__init__()

        self.N = N  # default is 8 for JPEG
        self.fre_len = N * N
        self.in_channal = in_channal
        self.out_channal =  N * N * in_channal
        # self.weight = torch.from_numpy(self.mk_coff(N = N)).float().unsqueeze(1)
        self.Ycbcr = nn.Conv2d(3, 3, 1, 1, bias=False) # can be moved
        trans_matrix = np.array([[0.299, 0.587, 0.114],
                                 [-0.169, -0.331, 0.5],
                                 [0.5, -0.419, -0.081]])
        trans_matrix = torch.from_numpy(trans_matrix).float().unsqueeze(
            2).unsqueeze(3)
        self.Ycbcr.weight.data = trans_matrix
        self.Ycbcr.weight.requires_grad = False


        # 3 H W -> N*N  H/N  W/N
        self.dct_conv = nn.Conv2d(self.in_channal, self.out_channal, N, N, bias=False, groups=self.in_channal)

        # 64 *1 * 8 * 8, from low frequency to high fre
        self.weight = torch.from_numpy(self.mk_coff(N = N, rearrange=True)).float().unsqueeze(1)
        # self.dct_conv = nn.Conv2d(1, self.out_channal, N, N, bias=False)
        self.dct_conv.weight.data = torch.cat([self.weight]*self.in_channal, dim=0) # 64 1 8 8
        self.dct_conv.weight.requires_grad = False



        # self.reDCT = nn.ConvTranspose2d(self.out_channal, 1, self.N,  self.N, bias = False)
        # self.reDCT.weight.data = self.weight






    def forward(self, x):
        # jpg = (jpg * self.std) + self.mean # 0-1
        '''
        x:  B C H W, 0-1. RGB
        YCbCr:  b c h w, YCBCR
        DCT: B C*64 H//8 W//8 ,   Y_L..Y_H  Cb_L...Cb_H   Cr_l...Cr_H

        '''
        # x = self.Ycbcr(x)  # b 3 h w
        dct = self.dct_conv(x)
        return dct

    def mk_coff(self, N = 8, rearrange = True):
        dct_weight = np.zeros((N*N, N, N))
        for k in range(N*N):
            u = k // N
            v = k % N
            for i in range(N):
                for j in range(N):
                    tmp1 = self.get_1d(i, u, N=N)
                    tmp2 = self.get_1d(j, v, N=N)
                    tmp = tmp1 * tmp2
                    tmp = tmp * self.get_c(u, N=N) * self.get_c(v, N=N)

                    dct_weight[k, i, j] += tmp
        if rearrange:
            dct_weight = self.get_order(dct_weight, N = N)  # from low frequency to high frequency
        return dct_weight # (N*N) * N * N

    def get_1d(self, ij, uv, N=8):
        result = math.cos(math.pi * uv * (ij + 0.5) / N)
        return result

    def get_c(self, u, N=8):
        if u == 0:
            return math.sqrt(1 / N)
        else:
            return math.sqrt(2 / N)

    def get_order(self, src_weight, N = 8):
        array_size = N * N
        # order_index = np.zeros((N, N))
        i = 0
        j = 0
        rearrange_weigth = src_weight.copy() # (N*N) * N * N
        for k in range(array_size - 1):
            if (i == 0 or i == N-1) and  j % 2 == 0:
                j += 1
            elif (j == 0 or j == N-1) and i % 2 == 1:
                i += 1
            elif (i + j) % 2 == 1:
                i += 1
                j -= 1
            elif (i + j) % 2 == 0:
                i -= 1
                j += 1
            index = i * N + j
            rearrange_weigth[k+1, ...] = src_weight[index, ...]
        return rearrange_weigth

class ReDCT(nn.Module):
    def __init__(self, N = 4, in_channal = 3):
        super(ReDCT, self).__init__()

        self.N = N  # default is 8 for JPEG
        self.in_channal = in_channal * N * N
        self.out_channal = in_channal
        self.fre_len = N * N

        self.weight = torch.from_numpy(self.mk_coff(N=N)).float().unsqueeze(1)


        self.reDCT = nn.ConvTranspose2d(self.in_channal, self.out_channal, self.N,  self.N, bias = False, groups=self.out_channal)
        self.reDCT.weight.data = torch.cat([self.weight]*self.out_channal, dim=0)
        self.reDCT.weight.requires_grad = False


    def forward(self, dct):
        '''
        IDCT  from DCT domain to pixle domain
        B C*64 H//8 W//8   ->   B C H W
        '''
        out = self.reDCT(dct)
        return out

    def mk_coff(self, N = 8, rearrange = True):
        dct_weight = np.zeros((N*N, N, N))
        for k in range(N*N):
            u = k // N
            v = k % N
            for i in range(N):
                for j in range(N):
                    tmp1 = self.get_1d(i, u, N=N)
                    tmp2 = self.get_1d(j, v, N=N)
                    tmp = tmp1 * tmp2
                    tmp = tmp * self.get_c(u, N=N) * self.get_c(v, N=N)

                    dct_weight[k, i, j] += tmp
        if rearrange:
            out_weight = self.get_order(dct_weight, N = N)  # from low frequency to high frequency
        return out_weight # (N*N) * N * N

    def get_1d(self, ij, uv, N=8):
        result = math.cos(math.pi * uv * (ij + 0.5) / N)
        return result

    def get_c(self, u, N=8):
        if u == 0:
            return math.sqrt(1 / N)
        else:
            return math.sqrt(2 / N)

    def get_order(self, src_weight, N = 8):
        array_size = N * N
        # order_index = np.zeros((N, N))
        i = 0
        j = 0
        rearrange_weigth = src_weight.copy() # (N*N) * N * N
        for k in range(array_size - 1):
            if (i == 0 or i == N-1) and  j % 2 == 0:
                j += 1
            elif (j == 0 or j == N-1) and i % 2 == 1:
                i += 1
            elif (i + j) % 2 == 1:
                i += 1
                j -= 1
            elif (i + j) % 2 == 0:
                i -= 1
                j += 1
            index = i * N + j
            rearrange_weigth[k+1, ...] = src_weight[index, ...]
        return rearrange_weigth

class CALayer(nn.Module):
    def __init__(self, channel=64, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//reduction, channel, 1, padding=0, bias=True),
            nn.Tanh()
        )

    def forward(self, x):
        y = self.avg_pool(x)

        y = self.conv_du(y)
        return x * y



class _NonLocalBlockND(nn.Module):
    """
    调用过程
    NONLocalBlock2D(in_channels=32),
    super(NONLocalBlock2D, self).__init__(in_channels,
            inter_channels=inter_channels,
            dimension=2, sub_sample=sub_sample,
            bn_layer=bn_layer)
    """

    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 dimension=2,
                 sub_sample = True,
                 bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            # 进行压缩得到channel个数
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0), bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                             out_channels=self.in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.phi = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c,  h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # [bs, c, w*h]
        g_x = g_x.permute(0, 2, 1) # b (h*w) c

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1) # b (h*w) c, querry

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1) # [bs, c, w*h], key

        f = torch.matmul(theta_x, phi_x)

        # print(f.shape, theta_x.shape, phi_x.shape)

        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z


class dwt_rep(nn.Module):
    def __init__(self, in_channal = 64, residual = False):
        super(dwt_rep, self).__init__()

        self.in_channal = in_channal
        self.manuan_out = in_channal * 8
        self.kernel_weigth = torch.tensor([[[1,1,1],[1,1,1],[1,1,1]],
                                       [[0,0,0],[1,1,1],[0,0,0]],
                                       [[0,1,0],[0,1,0],[0,1,0]],
                                       [[1,0,0],[0,1,0],[0,0,1]],
                                       [[0,0,1],[0,1,0],[1,0,0]],
                                       [[1,0,1],[0,-1,0],[1,0,1]],
                                       [[0,1,0],[-1,1,-1],[0,1,0]],
                                       [[1,0,1],[0,1,0],[1,0,1]]]).unsqueeze(1).float()

        self.kernel_weigth = torch.cat([self.kernel_weigth] * in_channal, dim=0)

        self.manual_conv = nn.Conv2d(in_channels=in_channal, out_channels=self.manuan_out, kernel_size=3, padding=1, groups=in_channal)
        self.manual_conv.weight.data = self.kernel_weigth
        self.manual_conv.weight.requires_grad = False


        self.conv_list = nn.ModuleList()
        for i in range(8):
            self.conv_list.append(nn.Conv2d(in_channels=in_channal, out_channels=in_channal, kernel_size=1, padding=0))

        self.relu = nn.PReLU()

    def forward(self, x):
        tmp = self.manual_conv(x)
        b,c,h,w = tmp.shape
        tmp = tmp.reshape(b,  c//8, 8, h, w)
        out = []

        for i in range(8):
            out.append(self.conv_list[i](tmp[:,:,i,:,:]))



        tmp = torch.stack(out, dim=0)
        tmp = torch.sum(tmp, dim=0, keepdim=False)

        out = self.relu(tmp + x)




        return out


class InverShift(nn.Conv2d):
    '''
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        x = (x - mean) / std
    '''
    def __init__(
        self, rgb_range=1.0,
        rgb_mean=(0.485, 0.456, 0.406), rgb_std=(0.229, 0.224, 0.225)):



        super(InverShift, self).__init__(3, 3, kernel_size=1, bias=True)
        std = torch.Tensor(rgb_std)
        mean = torch.Tensor(rgb_mean)

        self.weight.data = torch.eye(3).view(3, 3, 1, 1) * std.view(3, 1, 1, 1)
        self.bias.data = mean.view(3,)

        for p in self.parameters():
            p.requires_grad = False

class YcbcrShift(nn.Conv2d):
    '''

        RGB2Ycbcr
    '''
    def __init__(
        self):

        super(YcbcrShift, self).__init__(3, 3, kernel_size=1)

        trans_matrix = torch.tensor([[0.299, 0.587, 0.114],

                                         [-0.169, -0.331, 0.5],

                                         [0.5, -0.419, -0.081]])


        self.weight.data = trans_matrix.float().unsqueeze(
            2).unsqueeze(3)
        # self.bias.data = mean.view(3,)

        for p in self.parameters():
            p.requires_grad = False



def get_coffe_dct_n(N=3):
    '''

    :param N:
    :return: (N*N) * N * N
    '''
    def mk_coff(self, N = 8):
        dct_weight = np.zeros((N*N, N, N))
        for k in range(N*N):
            u = k // N
            v = k % N
            for i in range(N):
                for j in range(N):
                    tmp1 = get_1d(i, u, N=N)
                    tmp2 = get_1d(j, v, N=N)
                    tmp = tmp1 * tmp2
                    tmp = tmp * get_c(u, N=N) * get_c(v, N=N)

                    dct_weight[k, i, j] += tmp
        return dct_weight # (N*N) * N * N

    def get_1d(self, ij, uv, N=8):
        result = math.cos(math.pi * uv * (ij + 0.5) / N)
        return result

    def get_c(self, u, N=8):
        if u == 0:
            return math.sqrt(1 / N)
        else:
            return math.sqrt(2 / N)
    return mk_coff(N=N)


class DWTForward(nn.Module):
    '''
    input  c h w, out  (4*c)  * (h//2) * (w // 2)
    '''
    def __init__(self):
        super(DWTForward, self).__init__()
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
        filts = np.stack([ll[None, ::-1, ::-1], lh[None, ::-1, ::-1],
                          hl[None, ::-1, ::-1], hh[None, ::-1, ::-1]],
                         axis=0)
        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad=False)

    def forward(self, x):
        C = x.shape[1]
        filters = torch.cat([self.weight, ] * C, dim=0)
        y = F.conv2d(x, filters, groups=C, stride=2)
        return y


class DWTInverse(nn.Module):
    def __init__(self):
        super(DWTInverse, self).__init__()
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
        filts = np.stack([ll[None, ::-1, ::-1], lh[None, ::-1, ::-1],
                          hl[None, ::-1, ::-1], hh[None, ::-1, ::-1]],
                         axis=0)
        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad=False)

    def forward(self, x):
        C = int(x.shape[1] / 4)
        filters = torch.cat([self.weight, ] * C, dim=0)
        y = F.conv_transpose2d(x, filters, groups=C, stride=2)
        return y




if __name__ == '__main__':

    x = torch.rand(2,64,112,112)


    model = _NonLocalBlockND(in_channels=64)


    model  = DCT(N=3, in_channal=3)

    for i in range(3):
        for j in range(3):
            plt.subplot(3,3,i*3+j+1)
            plt.imshow(model.weight[i*3+j,0,...], cmap='gray')
    plt.show()
