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

        # self.model = CNN()
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
        self.acc_lst_zero_trn = manager.list()
        self.acc_lst_rest_trn = manager.list()
        self.b_size_lst_total_trn = manager.list()
        self.b_size_lst_zero_trn = manager.list()
        self.b_size_lst_rest_trn = manager.list()

        self.loss_lst_val = manager.list()

        self.acc_lst_total_val = manager.list()
        self.acc_lst_zero_val = manager.list()
        self.acc_lst_rest_val = manager.list()
        self.b_size_lst_total_val = manager.list()
        self.b_size_lst_zero_val = manager.list()
        self.b_size_lst_rest_val = manager.list()

        self.avg_loss_lst_trn = manager.list()
        self.avg_acc_lst_total_trn = manager.list()
        self.avg_acc_lst_zero_trn = manager.list()
        self.avg_acc_lst_rest_trn = manager.list()

        self.avg_loss_lst_val = manager.list()
        self.avg_acc_lst_total_val = manager.list()
        self.avg_acc_lst_zero_val = manager.list()
        self.avg_acc_lst_rest_val = manager.list()

        self.num4epoch = 0
        self.save_range = save_range
        self.total_reward_lst = manager.list()


    def flush_lst(self):
        self.loss_lst_trn = manager.list()

        self.acc_lst_total_trn = manager.list()
        self.acc_lst_zero_trn = manager.list()
        self.acc_lst_rest_trn = manager.list()
        self.b_size_lst_total_trn = manager.list()
        self.b_size_lst_zero_trn = manager.list()
        self.b_size_lst_rest_trn = manager.list()

        self.loss_lst_val = manager.list()

        self.acc_lst_total_val = manager.list()
        self.acc_lst_zero_val = manager.list()
        self.acc_lst_rest_val = manager.list()
        self.b_size_lst_total_val = manager.list()
        self.b_size_lst_zero_val = manager.list()
        self.b_size_lst_rest_val = manager.list()

        print('flushing lst done on model level')

    def forward(self, x):
        output = self.model(x.float())

        return output

    def crossentropy_loss(self, pred, label):

        correct = torch.argmax(pred, axis=1)
        label_zero = label[label == 0]
        label_rest = label[label != 0]

        label_zero_mask = label == 0
        label_rest_mask = label != 0

        correct_mask = correct == label
        mixed_mask_zero = label_zero_mask * correct_mask
        mixed_mask_rest = label_rest_mask * correct_mask

        acc_total = len(correct[correct == label])
        acc_zero = len(correct[mixed_mask_zero])
        acc_rest = len(correct[mixed_mask_rest])

        crossentropy_loss = nn.CrossEntropyLoss()
        cross_loss = crossentropy_loss(pred, label)

        b_size_total = len(label)
        b_size_zero = len(label_zero)
        b_size_rest = len(label_rest)

        return cross_loss, acc_total, acc_zero, acc_rest, b_size_total, b_size_zero,b_size_rest

    def training_step(self, train_batch, batch_idx):
        b_input, b_label = train_batch
        logits = self(b_input)

        cross_loss, acc_total, acc_zero, acc_rest, b_size_total, b_size_zero, b_size_rest = self.crossentropy_loss(pred=logits, label=b_label)

        self.loss_lst_trn.append(float(cross_loss.clone().detach().item()))
        self.acc_lst_total_trn.append(acc_total)
        self.acc_lst_zero_trn.append(acc_zero)
        self.acc_lst_rest_trn.append(acc_rest)
        self.b_size_lst_total_trn.append(b_size_total)
        self.b_size_lst_zero_trn.append(b_size_zero)
        self.b_size_lst_rest_trn.append(b_size_rest)

        return cross_loss

    def validation_step(self, val_batch, batch_idx):
        val_b_input, val_b_label = val_batch
        logits = self(val_b_input)

        cross_loss, acc_total, acc_zero, acc_rest, b_size_total, b_size_zero, b_size_rest = self.crossentropy_loss(pred=logits, label=val_b_label)

        self.loss_lst_val.append(float(cross_loss.clone().detach().item()))
        self.acc_lst_total_val.append(acc_total)
        self.acc_lst_zero_val.append(acc_zero)
        self.acc_lst_rest_val.append(acc_rest)
        self.b_size_lst_total_val.append(b_size_total)
        self.b_size_lst_zero_val.append(b_size_zero)
        self.b_size_lst_rest_val.append(b_size_rest)

        return cross_loss

    def validation_epoch_end(self,validiation_step_outputs):

        self.avg_loss_lst_trn.append(np.mean(self.loss_lst_trn))
        self.avg_loss_lst_val.append(np.mean(self.loss_lst_val))

        self.avg_acc_lst_total_trn.append(np.sum(self.acc_lst_total_trn)/np.sum(self.b_size_lst_total_trn))
        self.avg_acc_lst_total_val.append(np.sum(self.acc_lst_total_val)/np.sum(self.b_size_lst_total_val))

        self.avg_acc_lst_zero_trn.append(np.sum(self.acc_lst_zero_trn)/np.sum(self.b_size_lst_zero_trn))
        self.avg_acc_lst_zero_val.append(np.sum(self.acc_lst_zero_val)/np.sum(self.b_size_lst_zero_val))

        self.avg_acc_lst_rest_trn.append(np.sum(self.acc_lst_rest_trn)/np.sum(self.b_size_lst_rest_trn))
        self.avg_acc_lst_rest_val.append(np.sum(self.acc_lst_rest_val) / np.sum(self.b_size_lst_rest_val))

        self.total_reward_lst.append(np.sum(self.acc_lst_rest_val) / np.sum(self.b_size_lst_rest_val)+
            np.sum(self.acc_lst_zero_val)/np.sum(self.b_size_lst_zero_val) * self.weight4zero)

        if len(self.avg_loss_lst_val)>6:
            print(f'avg error of last 5 loss_val is : {cal_avg_error(self.loss_lst_val[-5:], self.loss_lst_val[-6:-1])} while stop_threshold : {self.stop_threshold} ')


        if self.num4epoch % self.save_range == 0:

            saving_name = mk_name(self.num4epoch,ls_trn=round(self.avg_loss_lst_val[-1],2),acc_val=round(self.avg_acc_lst_total_val[-1],2),
                                  acc_zr_val=round(self.avg_acc_lst_zero_val[-1],2),acc_rs_val=round(self.avg_acc_lst_rest_val[-1],2),totl_rwd=round(self.total_reward_lst[-1],2))

            #print(self.avg_acc_lst_total_trn,self.avg_acc_lst_zero_trn,self.avg_acc_lst_rest_trn)

            lst2csv(save_dir=self.save_dir,save_name=saving_name,loss_trn=list(self.avg_loss_lst_trn),
                    acc_total_trn = list(self.avg_acc_lst_total_trn),acc_zero_trn=list(self.avg_acc_lst_zero_trn),
                    acc_rest_trn = list(self.avg_acc_lst_rest_trn),loss_val= list(self.avg_loss_lst_val),
                    acc_total_val= list(self.avg_acc_lst_total_val),acc_zero_val = list(self.avg_acc_lst_zero_val),
                    acc_rest_val = list(self.avg_acc_lst_rest_val),total_reward=list(self.total_reward_lst))

            fig = plt.figure()
            ax1 = fig.add_subplot(2, 5, 1)
            ax1.plot(range(len(self.avg_loss_lst_trn)), self.avg_loss_lst_trn)
            ax1.set_title('train loss')
            ax2 = fig.add_subplot(2, 5, 2)
            ax2.plot(range(len(self.avg_acc_lst_total_trn)), self.avg_acc_lst_total_trn)
            ax2.set_title('train total acc')
            ax3 = fig.add_subplot(2, 5, 3)
            ax3.plot(range(len(self.avg_acc_lst_zero_trn)),self.avg_acc_lst_zero_trn)
            ax3.set_title('train zero acc')
            ax4 = fig.add_subplot(2, 5, 4)
            ax4.plot(range(len(self.avg_acc_lst_rest_trn)), self.avg_acc_lst_rest_trn)
            ax4.set_title('train rest acc')

            ax5 = fig.add_subplot(2, 5, 6)
            ax5.plot(range(len(self.avg_loss_lst_val)), self.avg_loss_lst_val)
            ax5.set_title('val loss')
            ax6 = fig.add_subplot(2, 5, 7)
            ax6.plot(range(len(self.avg_acc_lst_total_val)), self.avg_acc_lst_total_val)
            ax6.set_title('val total acc')
            ax7 = fig.add_subplot(2, 5, 8)
            ax7.plot(range(len(self.avg_acc_lst_zero_val)), self.avg_acc_lst_zero_val)
            ax7.set_title('val zero acc')
            ax8 = fig.add_subplot(2, 5, 9)
            ax8.plot(range(len(self.avg_acc_lst_rest_val)), self.avg_acc_lst_rest_val)
            ax8.set_title('val rest acc')
            ax9 = fig.add_subplot(2, 5, 10)
            ax9.plot(range(len(self.total_reward_lst)), self.total_reward_lst)
            ax9.set_title('total reward')


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

        self.train_inputs = torch.tensor(total_tdata).clone().detach().unsqueeze(1)
        self.train_labels = torch.tensor(total_tlabel).clone().detach()
        print('theta shape of train input,label is : ', self.train_inputs.shape, self.train_labels.shape)
        print('theta spliting train data done')

        self.val_inputs = (torch.tensor(val_data)).clone().detach().unsqueeze(1)
        self.val_labels = (torch.tensor(val_label)).clone().detach()
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
    def __init__(self,save_load_path,Weigh4zero,bs_trn,bs_val,noise_ratio,split_ratio,save_range=10,Max_Epoch=1000,Stop_Threshold=0.01):

        self.save_load_path = save_load_path
        self.Weight4Zero = Weigh4zero
        self.bs_trn = bs_trn
        self.bs_val = bs_val
        self.Max_Epoch = Max_Epoch
        self.Stop_Threshold = Stop_Threshold
        self.noise_ratio = noise_ratio
        self.split_ratio = split_ratio
        self.save_range = save_range

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

        RL_train_label_zero = RL_train_label[RL_train_label == 0]
        RL_train_label_rest = RL_train_label[RL_train_label != 0]

        RL_train_data_zero = RL_train_data[RL_train_label == 0]
        RL_train_data_rest = RL_train_data[RL_train_label != 0]

        RL_train_data_zero_little = mk_noisy_data(raw_data=RL_train_data_zero, split_ratio=self.split_ratio,
                                                  noise_ratio=self.noise_ratio,way='pureonly')
        RL_train_label_zero_little = RL_train_label_zero[:self.split_ratio]

        Total_train_data = np.vstack((RL_train_data_rest, RL_train_data_zero_little))
        Total_train_label = np.concatenate((RL_train_label_rest, RL_train_label_zero_little))

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

            if len(model.avg_loss_lst_val) > 11:
                if cal_avg_error(model.avg_loss_lst_val[-10:],model.avg_loss_lst_val[-11:-1]) < self.Stop_Threshold or i >= 50:
                    break

        torch.save(model,self.save_dir+str(i)+'.pt')

        del model
        del dm

        print(f'Training with noise : {self.noise_ratio}, split : {self.split_ratio}, weight4zero : {self.Weight4Zero} complete')


if __name__ == '__main__':
    save_load_path = '/home/emeraldsword1423/dir_4_naive_clsf_lossval_ver_purever/'

    bs_trn = 1024
    bs_val = 1024
    Max_Epoch = 1
    Stop_Threshold = 0.0001
    save_range = 3

    noise_ratio_lst = [1]
    split_ratio_lst = [10,20,40,80,160,320,640,1280,2560,5120]
    weight4zero_lst = [1,10,100,1000]

    for noise_ratio in noise_ratio_lst:
        for split_ratio in split_ratio_lst:
            for Weight4Zero in weight4zero_lst:
                Doit = test_start(save_load_path=save_load_path, Weigh4zero=Weight4Zero, bs_trn=bs_trn, bs_val=bs_val,
                              noise_ratio=noise_ratio, split_ratio=split_ratio, save_range=10, Max_Epoch=Max_Epoch,
                              Stop_Threshold=Stop_Threshold).test_start()







'home/emeraldsword1423/splt_rto_1000_nois_rto_1_W4Z_1_stp_thrs_0.01_/ls_trn2.14_acc_val0.28_acc_zr_val0.0_acc_rs_val0.31_totl_rwd0.31_0_.csv'
'~/splt_rto_1000_nois_rto_1_W4Z_1_stp_thrs_0.01_/'









