import torch
import json
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
import matplotlib.pyplot as plt
import copy
import pandas as pd
import numpy as np
import random
import time
import datetime
import os
import argparse
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import multiprocessing
from glob import glob
from torch.distributions import Categorical
from torchvision import models
from torchvision.datasets import MNIST
import gym
import csv
from CALAVG import cal_avg_error
from MODELS import CNN
from MK_NOISED_DATA import mk_noisy_data
from save_funcs import mk_name,lst2csv,createDirectory

# config 클래스
class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setattr__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)


manager = multiprocessing.Manager()

class Prediction_lit(pl.LightningModule):
    def __init__(self,save_dir,weigh4zero,save_range,stop_threshold):
        super().__init__()

        #self.model = CNN()
        self.model = models.resnet18(pretrained=False)

        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #Change the output layer to output 10 classes instead of 1000 classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)

        self.save_dir = save_dir

        self.weight4zero = weigh4zero
        self.stop_threshold = stop_threshold

        self.loss_lst_trn = manager.list()
        self.acc_lst_total_trn = manager.list()
        self.loss_lst_val = manager.list()
        self.acc_lst_total_val = manager.list()

        self.avg_loss_lst_trn = manager.list()
        self.avg_acc_lst_total_trn = manager.list()
        self.avg_loss_lst_val = manager.list()
        self.avg_acc_lst_total_val = manager.list()

        self.num4epoch = 0
        self.save_range = save_range
        self.total_reward_lst = manager.list()

    def flush_lst(self):
        self.loss_lst_trn = manager.list()
        self.acc_lst_total_trn = manager.list()

        self.loss_lst_val = manager.list()
        self.acc_lst_total_val = manager.list()

        print('flushing lst done on model level')

    def forward(self, x):
        output = self.model(x.float())

        return output

    def crossentropy_loss(self, pred, label):

        correct = torch.argmax(pred, axis=1)

        correct_mask = correct == label
        acc_total = float(len(correct[correct == label])/len(label))

        crossentropy_loss = nn.CrossEntropyLoss()
        cross_loss = crossentropy_loss(pred, label)

        return cross_loss, acc_total

    def training_step(self, train_batch, batch_idx):
        b_input, b_label = train_batch
        logits = self(b_input)

        cross_loss, acc_total = self.crossentropy_loss(pred=logits, label=b_label)

        self.loss_lst_trn.append(float(cross_loss.clone().detach().item()))
        self.acc_lst_total_trn.append(acc_total)

        return cross_loss

    def validation_step(self, val_batch, batch_idx):
        val_b_input, val_b_label = val_batch
        logits = self(val_b_input)

        cross_loss, acc_total = self.crossentropy_loss(pred=logits, label=val_b_label)

        self.loss_lst_val.append(float(cross_loss.clone().detach().item()))
        self.acc_lst_total_val.append(acc_total)

        return cross_loss

    def validation_epoch_end(self,validiation_step_outputs):

        self.avg_loss_lst_trn.append(np.mean(self.loss_lst_trn))
        self.avg_loss_lst_val.append(np.mean(self.loss_lst_val))

        self.avg_acc_lst_total_trn.append(np.mean(self.acc_lst_total_trn))
        self.avg_acc_lst_total_val.append(np.mean(self.acc_lst_total_val))

        #self.total_reward_lst.append(np.sum(self.acc_lst_rest_val) / np.sum(self.b_size_lst_rest_val)+
        #    np.sum(self.acc_lst_zero_val)/np.sum(self.b_size_lst_zero_val) * self.weight4zero)

        if len(self.avg_loss_lst_val)>6:
            print(f'avg error of last 5 loss_val is : {cal_avg_error(self.loss_lst_val[-5:], self.loss_lst_val[-6:-1])} while stop_threshold : {self.stop_threshold} ')


        if self.num4epoch % self.save_range == 0:

            saving_name = mk_name(self.num4epoch,ls_val=round(self.avg_loss_lst_val[-1],2),acc_val=round(self.avg_acc_lst_total_val[-1],2))

            #print(self.avg_acc_lst_total_trn,self.avg_acc_lst_zero_trn,self.avg_acc_lst_rest_trn)

            lst2csv(save_dir=self.save_dir,save_name=saving_name,loss_trn=list(self.avg_loss_lst_trn),
                    acc_total_trn = list(self.avg_acc_lst_total_trn),
                    loss_val= list(self.avg_loss_lst_val),
                    acc_total_val= list(self.avg_acc_lst_total_val),
                    )

            fig = plt.figure()
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.plot(range(len(self.avg_loss_lst_trn)), self.avg_loss_lst_trn)
            ax1.set_title('train loss')
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.plot(range(len(self.avg_acc_lst_total_trn)), self.avg_acc_lst_total_trn)
            ax2.set_title('train total acc')

            ax5 = fig.add_subplot(2, 2, 3)
            ax5.plot(range(len(self.avg_loss_lst_val)), self.avg_loss_lst_val)
            ax5.set_title('val loss')
            ax6 = fig.add_subplot(2, 2, 4)
            ax6.plot(range(len(self.avg_acc_lst_total_val)), self.avg_acc_lst_total_val)
            ax6.set_title('val total acc')

            plt.savefig(self.save_dir+saving_name+'.png', dpi=300)
            print('saving plot complete!')
            plt.close()

        self.num4epoch +=1
        self.flush_lst()

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(),
                          lr=4e-6,  # 학습률
                          eps=1e-9  # 0으로 나누는 것을 방지하기 위한 epsilon 값
                          )
        return optimizer

class datamodule(pl.LightningDataModule):
    def __init__(self,total_tdata,total_tlabel,val_data,val_label,batch_size=1,batch_size_val=1):
        super().__init__()

        self.batch_size = batch_size
        self.batch_size_val = batch_size_val
        self.number_of_epoch = 0

        print('theta setup start....')

        self.train_inputs = torch.tensor(total_tdata).unsqueeze(1)
        self.train_labels = torch.tensor(total_tlabel)
        print('theta shape of train input,label is : ', self.train_inputs.shape, self.train_labels.shape)
        print('theta spliting train data done')

        self.val_inputs = (torch.tensor(val_data)).unsqueeze(1)
        self.val_labels = (torch.tensor(val_label))
        print('theta spliting validation data done')

    def prepare_data(self, stage=None):
        pass

    def flush_data(self):
        self.train_inputs = 0
        self.train_labels = 0
        self.val_inputs = 0
        self.val_labels = 0
        pass
        #print('flushing data done.')

    def setup(self, stage=None):

        if stage == 'fit' or stage is None:
            print('theta stage is ', stage)

        if stage == 'test' or stage is None:
            pass

    def train_dataloader(self):
        print('train_dataloading.......')
        train_data = TensorDataset(self.train_inputs, self.train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size, num_workers=32)
        print('train_dataloading done....')

        return train_dataloader

    def val_dataloader(self):
        validation_data = TensorDataset(self.val_inputs, self.val_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=self.batch_size_val,
                                           num_workers=32)
        return validation_dataloader

    def test_dataloader(self):
        pass
        # test_data = TensorDataset(self.test_inputs, self.test_labels)
        # test_sampler = RandomSampler(test_data)
        # test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.batch_size, num_workers=4)

        # return test_dataloader

class test_start():
    def __init__(self,wayofdata,Max_Stop_Num,save_load_path,Weigh4zero,bs_trn,bs_val,noise_ratio,split_ratio,save_range=10,Max_Epoch=1000,Stop_Threshold=0.01):

        self.save_load_path = save_load_path
        self.Weight4Zero = Weigh4zero
        self.bs_trn = bs_trn
        self.bs_val = bs_val
        self.Max_Epoch = Max_Epoch
        self.Stop_Threshold = Stop_Threshold
        self.noise_ratio = noise_ratio
        self.split_ratio = split_ratio
        self.save_range = save_range
        self.wayofdata = wayofdata
        self.Max_Stop_num = Max_Stop_Num

        self.arg_lst = ['splt_rto', str(self.split_ratio), 'nois_rto', str(self.noise_ratio), 'W4Z', str(self.Weight4Zero),'stp_thrs',str(self.Stop_Threshold)]
        self.save_dir = self.save_load_path + mk_name(*self.arg_lst) + '/'

        try:
            createDirectory(self.save_dir[:-1])
            print(f'making dir {self.save_dir[:-1]} complete successfully')
        except:
            print(f'already {self.save_dir} folder exists')


    def test_start(self):

        RL_train_dataset = MNIST(self.save_load_path, train=True, download=True)
        RL_val_dataset = MNIST(self.save_load_path, train=False, download=True)

        RL_train_data = RL_train_dataset.data.numpy()
        RL_train_label = RL_train_dataset.targets.numpy()

        if self.wayofdata == 'pureonly':
            Total_train_data = mk_noisy_data(raw_data=RL_train_data,way=self.wayofdata,split_ratio=self.split_ratio,noise_ratio=self.noise_ratio)
            Total_train_label = RL_train_label[:self.split_ratio]
        elif self.wayofdata == 'sum':
            Total_train_data = mk_noisy_data(raw_data=RL_train_data, way=self.wayofdata, split_ratio=self.split_ratio,
                                             noise_ratio=self.noise_ratio)
            Total_train_label = RL_train_label[:]


        RL_val_data = RL_val_dataset.data.numpy()
        RL_val_label = RL_val_dataset.targets.numpy()

        print('spliting train data done')

        del RL_train_dataset
        del RL_train_data
        del RL_val_dataset

        model = Prediction_lit(save_dir=self.save_dir,weigh4zero=self.Weight4Zero,save_range=self.save_range,stop_threshold=self.Stop_Threshold)
        dm = datamodule(batch_size=self.bs_trn, batch_size_val=self.bs_val,total_tdata=Total_train_data,
        total_tlabel=Total_train_label,val_data=RL_val_data,val_label=RL_val_label)

        for i in range(10000):
            trainer = pl.Trainer(gpus=1,accelerator='dp',max_epochs=self.Max_Epoch, checkpoint_callback=False, logger=False,
                                 num_sanity_val_steps=0, weights_summary=None)

            trainer.fit(model, dm)

            if len(model.avg_acc_lst_total_val) > 11:
                if cal_avg_error(model.avg_acc_lst_total_val[-10:],model.avg_acc_lst_total_val[-11:-1]) < self.Stop_Threshold or i >= self.Max_Stop_num:
                    break

        try:
            trainer.save(self.save_dir+str(i)+'.ckpt')
            print('saving latest model sucess')
        except:
            print('saving model failed')

        del model
        del dm

        print(f'Training with noise : {self.noise_ratio}, split : {self.split_ratio}, weight4zero : {self.Weight4Zero} complete')


if __name__ == '__main__':
    save_load_path = '/home/emeraldsword1423/dir_4_naive_clsf_total_acc0.0001ver_sum_pretraineFALSE/'

    bs_trn = 1024
    bs_val = 1024
    Max_Epoch = 1
    Stop_Threshold = 0.0001
    save_range = 10
    Max_stop_num = 200
    wayofdata = 'sum'

    noise_ratio_lst = [1,2,3]
    split_ratio_lst = [6000,12000,18000,24000,30000,36000,42000,48000,54000]
    weight4zero_lst = [1]

    for noise_ratio in noise_ratio_lst:
        for split_ratio in split_ratio_lst:
            for Weight4Zero in weight4zero_lst:
                Doit = test_start(wayofdata=wayofdata,Max_Stop_Num=Max_stop_num,save_load_path=save_load_path, Weigh4zero=Weight4Zero, bs_trn=bs_trn, bs_val=bs_val,
                              noise_ratio=noise_ratio, split_ratio=split_ratio, save_range=save_range, Max_Epoch=Max_Epoch,
                              Stop_Threshold=Stop_Threshold).test_start()







'home/emeraldsword1423/splt_rto_1000_nois_rto_1_W4Z_1_stp_thrs_0.01_/ls_trn2.14_acc_val0.28_acc_zr_val0.0_acc_rs_val0.31_totl_rwd0.31_0_.csv'
'~/splt_rto_1000_nois_rto_1_W4Z_1_stp_thrs_0.01_/'









