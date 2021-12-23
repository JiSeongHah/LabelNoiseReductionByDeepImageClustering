import csv
import pickle

import torch
import json
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW, Adam
import matplotlib.pyplot as plt
import copy
import pandas as pd
import numpy as np
import random
import time
import datetime
import os

import torch.nn.functional as F
import torch.nn as nn



import multiprocessing
from glob import glob
import pytorch_lightning as pl
from tifffile import tifffile
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms

import random


class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # 첫번째층
        # ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 두번째층
        # ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 전결합층 7x7x64 inputs -> 10 outputs
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)

        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)  # 전결합층을 위해서 Flatten
        out = self.fc(out)
        return out




class BasicBlock(nn.Module):
    # mul은 추후 ResNet18, 34, 50, 101, 152등 구조 생성에 사용됨
    mul = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()

        # stride를 통해 너비와 높이 조정
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)

        # stride = 1, padding = 1이므로, 너비와 높이는 항시 유지됨
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        # x를 그대로 더해주기 위함
        self.shortcut = nn.Sequential()

        # 만약 size가 안맞아 합연산이 불가하다면, 연산 가능하도록 모양을 맞춰줌
        if stride != 1:  # x와
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)  # 필요에 따라 layer를 Skip
        out = F.relu(out)
        return out


class BottleNeck(nn.Module):
    # 논문의 구조를 참고하여 mul 값은 4로 지정, 즉, 64 -> 256
    mul = 4

    def __init__(self, in_planes, out_planes, stride=1):
        super(BottleNeck, self).__init__()



        # 첫 Convolution은 너비와 높이 downsampling
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.conv3 = nn.Conv2d(out_planes, out_planes * self.mul, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes * self.mul)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != out_planes * self.mul:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes * self.mul, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes * self.mul)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        # RGB 3개채널에서 64개의 Kernel 사용 (논문 참고)
        self.in_planes = 64

        # Resnet 논문 구조의 conv1 파트 그대로 구현
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Basic Resiudal Block일 경우 그대로, BottleNeck일 경우 4를 곱한다.
        self.linear = nn.Linear(512 * block.mul, num_classes)

    # 다양한 Architecture 생성을 위해 make_layer로 Sequential 생성
    def make_layer(self, block, out_planes, num_blocks, stride):
        # layer 앞부분에서만 크기를 절반으로 줄이므로, 아래와 같은 구조
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            layers.append(block(self.in_planes, out_planes, strides[i]))
            self.in_planes = block.mul * out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.maxpool1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out




class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
        super(ResnetGenerator, self).__init__()

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=False),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=False)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=False),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""

        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting path
        self.enc1_1 = CBR2d(in_channels=3, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

        # Expansive path
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)
        # print(pool3.size())
        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)

        return x




def cal_avg_error(x,y):
    x = np.asarray(x)
    y = np.asarray(y)

    error = np.mean(abs(x-y))

    return error


def label_to_one_hot_label(
        labels: torch.Tensor,
        num_classes: int,
        eps: float = 1e-6,
) -> torch.Tensor:

    shape = labels.shape
#    print(shape[0],shape[1:])

    one_hot = torch.zeros((shape[0], num_classes) + shape[1:])
#    print(one_hot.size())


    one_hoted_label = one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps
    return one_hoted_label

def mk_name(*args,**name_value_dict):
    total_name = ''
    additional_arg = ''

    for arg in args:
        additional_arg += (str(arg)+'_')

    for name_value in name_value_dict.items():
        name = name_value[0]
        value = name_value[1]
        total_name += (str(name)+str(value)+'_')

    total_name += additional_arg[:-1]

    return total_name

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f'making {str(directory)} complete successfully!')
    except OSError:
        print("Error: Failed to create the directory.")

def fill_zero(raw_num):
    return str(raw_num.zfill(3))

def clip_max(raw_num,max_num):
    if raw_num>max_num:
        return max_num
    else:
        return raw_num


class my_dataset(Dataset):
    def __init__(self,dir_lst,normalize=False):
        self.dir_lst = dir_lst


        self.testnum = 0

        self.channel_test = transforms.ToTensor()

        if normalize == True:
            self.transforms = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                                  ])
        else:
            self.transforms = transforms.Compose([transforms.ToTensor()
                                                  ])


    def __len__(self):
        return len(self.dir_lst)

    def __getitem__(self, idx):

        rand_dir_element = self.dir_lst[idx]


        # print('11')

        try:
            #print(rand_dir_element)
            img_nparr = io.imread(rand_dir_element[0]).astype(np.float32)

            # print(f'shape of img is : {img_nparr.shape}')
            # print('22')
            img_tensor = torch.from_numpy(img_nparr)
            # print(f'after tersoring, size of imge is : {img_tensor.size()}')

            # print('33')
            label_nparr = io.imread(rand_dir_element[1]).astype(np.int64)
            # print('44')
            label_tensor = torch.from_numpy(label_nparr).unsqueeze(0)
            # print('55')
            label_tensor = label_to_one_hot_label(labels=label_tensor,num_classes=121).squeeze(0)

            if img_tensor.size() != (150,128,128):
                img_tensor = img_tensor[:150,:,:]

            #print(f'size of img : {img_tensor.size()} and size of label is : {label_tensor.size()}')

            #self.dir4remove.remove(rand_dir_element)
        except:
            print('found file which is failed to load')
            print('found file which is failed to load')
            print('found file which is failed to load')
            print(rand_dir_element)
            print('found file which is failed to load')
            print('found file which is failed to load')
            print('found file which is failed to load')

        return img_tensor,label_tensor


class my_test_dataset(Dataset):
    def __init__(self,dir_lst,normalize=False):
        self.dir_lst = dir_lst

        self.testnum = 0

        self.channel_test = transforms.ToTensor()

        if normalize == True:
            self.transforms = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                                  ])
        else:
            self.transforms = transforms.Compose([transforms.ToTensor()
                                                  ])


    def __len__(self):
        return len(self.dir_lst)

    def __getitem__(self, idx):

        rand_dir_element = self.dir_lst[idx]

        f_name = rand_dir_element.split('/')[-1].split('.')[0]

        img_nparr = io.imread(rand_dir_element).astype(np.float32)

        # print(f'shape of img is : {img_nparr.shape}')
        # print('22')
        img_tensor = torch.from_numpy(img_nparr)

        if img_tensor.size() != (150,128,128):
            img_tensor = img_tensor[:150,:,:]



        return img_tensor,f_name



manager = multiprocessing.Manager()


class Datamodule_Project2(pl.LightningDataModule):
    def __init__(self, train_data_label_lst,val_data_label_lst,test_data_lst,train_b_size=32,val_b_size=32,num_workers=4):
        super().__init__()

        self.train_b_size = train_b_size
        self.val_b_size = val_b_size

        self.train_data_label_lst = train_data_label_lst
        self.val_data_label_lst = val_data_label_lst
        self.test_data_lst = test_data_lst

        self.num_workers = num_workers


    def prepare_data(self, stage=None):
        pass

    def flush_data(self):
        print('flushing data done.')

    def setup(self, stage=None):

        if stage == 'fit' or stage is None:
            print('77777777777777777')
            self.my_train_dataset = my_dataset(dir_lst=self.train_data_label_lst)
            # print('88888888888888888')
            self.my_val_dataset = my_dataset(dir_lst=self.val_data_label_lst)
            print('9999999999999999')
            pass

        if stage == 'test' or stage is None:
            print('test data setting')
            self.my_test_dataset = my_test_dataset(dir_lst=self.test_data_lst)

            print('test data setting complete')

    def train_dataloader(self):
        print('train_dataloading.......')

        train_dataloader = DataLoader(dataset=self.my_train_dataset, batch_size=self.train_b_size, shuffle=True,num_workers=self.num_workers)
        print('train_dataloading done....')

        return train_dataloader

    def val_dataloader(self):
        print('validation dataloader loading......')

        val_dataloader = DataLoader(dataset=self.my_val_dataset, batch_size=self.val_b_size, shuffle=False,num_workers=self.num_workers)
        print('validation dataloader loading done !!!')

        return val_dataloader

    def test_dataloader(self):
        print('test dataloader loading......')

        test_dataloader = DataLoader(dataset=self.my_test_dataset, batch_size=1, shuffle=False,
                                    num_workers=1)
        print('test dataloader loading done !!!')

        return test_dataloader


class Prediction_Lit_Project2(pl.LightningModule):
    def __init__(self,stop_threshold,save_range,model_and_plot_save_dir='./'):
        super().__init__()

        self.model = ResnetGenerator(input_nc=150,output_nc=121)

        self.loss_lst_trn = manager.list()
        self.acc_lst_trn = manager.list()
        self.loss_lst_val = manager.list()
        self.acc_lst_val = manager.list()

        self.avg_loss_lst_trn = manager.list()
        self.avg_loss_lst_val = manager.list()

        self.stop_threshold = stop_threshold

        self.model_and_plot_save_dir = model_and_plot_save_dir
        createDirectory(self.model_and_plot_save_dir)

        self.num4epoch = 0

        self.save_range = save_range

        self.correct_lst = manager.list()
        self.avg_correct_lst = manager.list()
        self.sig = nn.Sigmoid()

        self.lst4csv = [['Id', 'Predicted']]


    def flush_lst(self):
        self.loss_lst_trn = manager.list()
        self.acc_lst_trn = manager.list()
        self.loss_lst_val = manager.list()
        self.acc_lst_val = manager.list()
        self.correct_lst = manager.list()
        print('flushing lst done')

    def forward(self, x):

        output = self.model(x)


        return output

    def total_bce_loss(self, pred, label):

        loss_criterion = nn.BCEWithLogitsLoss()

        loss_result = loss_criterion(pred,label)

        return loss_result

    def training_step(self, train_batch, batch_idx):

        b_input, b_label = train_batch
        logits = self(b_input)


        loss_result = self.total_bce_loss(pred=logits,label=b_label)

        self.loss_lst_trn.append(float(loss_result.item()))


        return loss_result

    def validation_step(self, val_batch, batch_idx):
        val_b_input, val_b_label = val_batch

        logits = self(val_b_input)

        loss_result = self.total_bce_loss(pred=logits, label=val_b_label)



        index_logit = torch.argmax(self.sig(copy.deepcopy(logits)) ,dim=1).cpu()
        index_label = torch.argmax(copy.deepcopy(val_b_label),dim=1).cpu()

        print(index_logit.size(),index_label.size())
        correct= torch.mean(torch.eq(index_logit,index_label).float()).item()

        print(correct)

        self.correct_lst.append(correct)

        self.loss_lst_val.append(float(loss_result.item()))

        return loss_result

    def validation_epoch_end(self,validiation_step_outputs):

        self.avg_loss_lst_trn.append(np.mean(self.loss_lst_trn))
        self.avg_loss_lst_val.append(np.mean(self.loss_lst_val))
        self.avg_correct_lst.append(np.mean(self.correct_lst))

        self.flush_lst()

    def test_step(self,test_batch,batch_idx):

        test_b_input,data_name = test_batch

        data_name = data_name[0]

        logits = self(test_b_input)
        #print(logits.size())

        index_logits = torch.argmax(logits,dim=1)
        #print(index_logits.size())

        for Xcord in range(index_logits.size(1)):
            for Ycord in range(index_logits.size(2)):
                ID4csv = data_name +'_'+ fill_zero(str(Xcord))+'_'+fill_zero(str(Ycord))
                Pre4csv = clip_max(int(index_logits[0,Xcord,Ycord]),max_num=24)
                print(f'index is : {Pre4csv}')
                self.lst4csv.append([ID4csv,Pre4csv])


    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(),
                          lr=4e-6,  # 학습률
                          eps=1e-8  # 0으로 나누는 것을 방지하기 위한 epsilon 값
                          )
        return optimizer




save_dir = '/home/a286/hjs_dir1/Dir4Project2/'
# import os
#
# print(os.listdir(save_dir))
# createDirectory(save_dir)
#

#
# k= []
#
# train_data_path = save_dir +'/Training/'
# train_label_path = save_dir + '/Training/train-label/'
# train_dir_lst = os.listdir(train_data_path)
# train_dir_lst = [file for file in train_dir_lst if file not in ['train_json', 'train-label']]
# total_train_file_label_lst = []
#
#
#
# for big_dir in train_dir_lst:
#     for each_f_dir in glob(train_data_path+big_dir+'/'+big_dir+'/*'):
#         each_f_name = (each_f_dir.split('/')[-1]).split('.')[0]
#         each_label_f_dir = train_label_path+big_dir+'/'+big_dir+'/'+each_f_name+'_gt.tif'
#         total_train_file_label_lst.append([each_f_dir,each_label_f_dir])
#
#         k.append(np.max(tifffile.imread(each_label_f_dir)))
#         print('train lst appending number : ',len(k))
#         #print(k[-1])
#
# print('appending train_datas done')
#
#
# kk = []
#
# val_data_path = save_dir + '/Validation/'
# val_label_path = save_dir + '/Validation/val-label/'
# val_dir_lst = os.listdir(val_data_path)
# val_dir_lst = [file for file in val_dir_lst if file not in ['val_json', 'val-label','backupdir']]
# total_val_file_label_lst = []
#
#
# for big_dir in val_dir_lst:
#     for each_f_dir in glob(val_data_path+big_dir+'/'+big_dir+'/*'):
#         each_f_name = (each_f_dir.split('/')[-1]).split('.')[0]
#         each_label_f_dir = val_label_path+big_dir+'/'+each_f_name+'_gt.tif'
#         total_val_file_label_lst.append([each_f_dir,each_label_f_dir])
#
#         kk.append(np.max(tifffile.imread(each_label_f_dir)))
#         print('val lst appending number : ',len(kk))
#
#
#
# print('appending validation datas done')
# print(f'np.max(k) is : {np.max(k)}')
#
# dict4dump = dict()
# dict4dump['train_lst'] = total_train_file_label_lst
# dict4dump['val_lst'] = total_val_file_label_lst
#
# with open(save_dir+'total_data_label_lst.pickle','wb') as f:
#     pickle.dump(dict4dump,f)


test_data_path = save_dir + 'test/'

test_dir_lst = os.listdir(test_data_path)
test_dir_lst = [test_data_path+file for file in test_dir_lst if file not in ['val_json', 'val-label','backupdir']]


save_dir = '/home/a286/hjs_dir1/Dir4Project2/'

with open(save_dir+'total_data_label_lst.pickle','rb') as f:
    dict4load = pickle.load(f)

total_train_file_label_lst = dict4load['train_lst']
total_val_file_label_lst = dict4load['val_lst']

with open(save_dir+'dicttoremove.pickle','rb') as f:
    dict4load_remove = pickle.load(f)

total_train_to_remove = dict4load_remove['train_to_remove']
total_val_to_remove = dict4load_remove['val_to_remove']


for val_file_wrong in total_val_to_remove:
    total_val_file_label_lst.remove(val_file_wrong)
    print(f'val removing {val_file_wrong} complete')


print(len(total_train_file_label_lst),len(total_val_file_label_lst))



#
# lst4removetrn = copy.deepcopy(total_train_file_label_lst)
# lst4removeval = copy.deepcopy(total_val_file_label_lst)
# print(len(total_train_file_label_lst),len(total_val_file_label_lst))
#
# dm = my_dataset(dir_lst=total_train_file_label_lst)
# dt = DataLoader(dataset=dm, batch_size=1, num_workers=1)
#
# for i in dt:
#     if i[0] == 1 and i[1] == 1:
#         x = [list(kkk)[0] for kkk in i[2]]
#         print(x)
#         lst4removetrn.remove(x)
#         print(f'trn removing {x} complete')
#     pass
#
# del dm
# del dt
#
# dm = my_dataset(dir_lst=total_val_file_label_lst)
# dt = DataLoader(dataset=dm, batch_size=1, num_workers=1)
#
# for i in dt:
#     if i[0] == 1 and i[1] == 1:
#         x = [list(kkk)[0] for kkk in i[2]]
#         print(x)
#         lst4removeval.remove(x)
#         print(f'trn removing {x} complete')
#     pass
#
#
# dict4dump_remove = dict()
#
# dict4dump_remove['train_to_remove'] = lst4removetrn
# dict4dump_remove['val_to_remove'] = lst4removeval
#
# with open(save_dir+'dicttoremove.pickle','wb') as f:
#     pickle.dump(dict4dump_remove,f)


#
#
stop_threshold = 0.001
train_b_size = 200
val_b_size = 200
num_workers = 4
save_range = 1
model_and_plot_save_dir = save_dir +'resnet9blocks3/'

dm = Datamodule_Project2(train_data_label_lst=total_train_file_label_lst,val_data_label_lst=total_val_file_label_lst,test_data_lst=test_dir_lst,train_b_size=train_b_size,val_b_size=val_b_size,num_workers=num_workers)
model = Prediction_Lit_Project2(model_and_plot_save_dir=model_and_plot_save_dir,stop_threshold=stop_threshold,save_range=save_range)



for i in range(10000):
    trainer = pl.Trainer(gpus=[2, 3], strategy='dp', max_epochs=1, max_steps=77,enable_checkpointing=False, logger=False,
                         num_sanity_val_steps=3, enable_model_summary=None)
    trainer.fit(model, dm)
    if i%1 == 0:
        try:
            trainer.save_checkpoint(model_and_plot_save_dir+str(i)+'.ckpt')


            saving_name = mk_name(NumEpoch=str(i))

            fig = plt.figure()
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.plot(range(len(model.avg_loss_lst_trn)), model.avg_loss_lst_trn)
            ax1.set_title('train loss')
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.plot(range(len(model.avg_loss_lst_val)), model.avg_loss_lst_val)
            ax2.set_title('val loss')
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.plot(range(len(model.avg_correct_lst)), model.avg_correct_lst)
            ax3.set_title('avg acc')

            plt.savefig(model.model_and_plot_save_dir + saving_name + '.png', dpi=300)
            print('saving plot complete!')
            plt.close()

            print('saving model success')
            print('saving model success')
            print('saving model success')
            print('saving model success')
            time.sleep(5)

        except:
            print('saving model failed')
            print('saving model failed')
            print('saving model failed')
            print('saving model failed')
            print('saving model failed')
            time.sleep(5)




#
# dm = Datamodule_Project2(train_data_label_lst=total_train_file_label_lst,val_data_label_lst=total_val_file_label_lst,test_data_lst=test_dir_lst,train_b_size=train_b_size,val_b_size=val_b_size,num_workers=num_workers)
# model = Prediction_Lit_Project2(model_and_plot_save_dir=model_and_plot_save_dir,stop_threshold=stop_threshold,save_range=save_range)
# model = model.load_from_checkpoint(model_and_plot_save_dir+'8.ckpt',save_range=10,stop_threshold=10)
# trainer = pl.Trainer(gpus=[2, 3], strategy='dp', max_epochs=1, enable_checkpointing=False, logger=False,
#                          num_sanity_val_steps=0, enable_model_summary=None)
#
# trainer.test(model,dm)
#
# with open(model_and_plot_save_dir+'final3.csv','w',newline='') as f:
#     write = csv.writer(f)
#     write.writerows(model.lst4csv)
