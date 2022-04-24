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
import csv
from CALAVG import cal_avg_error
from MY_MODELS import BasicBlock4one, ResNet4one
from save_funcs import mk_name,lst2csv,createDirectory
from REINFORCE_DATAMODULES import datamodule_4REINFORCE1
from REINFORCE_INNER_MODELS import Prediction_lit_4REINFORCE1
from MK_NOISED_DATA import mk_noisy_data
from save_funcs import load_my_model



noise_ratio = 2
split_ratioLst = [int(5923*(2*i-1)/100) for i in range(1,12)]
wayofdata = 'pureonly'
trn_fle_down_path = '/home/a286/testtest2/'



RL_train_dataset = MNIST(trn_fle_down_path, train=True, download=True)
RL_val_dataset = MNIST(trn_fle_down_path, train=False, download=True)

RL_train_data = RL_train_dataset.data.numpy()
RL_train_label = RL_train_dataset.targets.numpy()

RL_train_label_zero = RL_train_label[RL_train_label == 0]
RL_train_label_rest = RL_train_label[RL_train_label != 0]

RL_train_data_zero = RL_train_data[RL_train_label == 0]
RL_td_rest = RL_train_data[RL_train_label != 0]
print(f'shape of b_input which is rest only is : {RL_td_rest.shape}')
RL_td_rest = torch.from_numpy(RL_td_rest).unsqueeze(1)
print(f'shape of b_input which is rest only is : {RL_td_rest.size()}')
RL_tl_rest = torch.ones_like(torch.from_numpy(RL_train_label_rest))


val_data = torch.from_numpy(RL_val_dataset.data.numpy()).clone().detach().unsqueeze(1)
val_label = torch.from_numpy(RL_val_dataset.targets.numpy()).clone().detach()
val_label_zero = val_label[val_label == 0]
val_label_rest = val_label[val_label != 0]

val_data_zero = val_data[val_label == 0]
val_data_rest = val_data[val_label != 0]
val_label_rest = torch.ones_like(val_label_rest)

total_val_data = torch.cat((val_data_rest, val_data_zero), dim=0)
total_val_label = torch.cat((val_label_rest, val_label_zero), dim=0)



for split_ratio in split_ratioLst:

    resultPure = []
    resultSum = []

    eachDir = trn_fle_down_path+str(split_ratio)+'/'

    createDirectory(eachDir)

    for i in range(10):

        RL_train_data_zero_littleSum = torch.from_numpy(mk_noisy_data(raw_data=RL_train_data_zero, noise_ratio=noise_ratio,
                                              split_ratio=split_ratio, way='sum')).unsqueeze(1)
        RL_train_label_zero_littleSum = torch.from_numpy(RL_train_label_zero[:])

        RL_train_data_zero_littlePure = torch.from_numpy(mk_noisy_data(raw_data=RL_train_data_zero, noise_ratio=noise_ratio,
                                                  split_ratio=split_ratio, way='pureonly')).unsqueeze(1)
        RL_train_label_zero_littlePure = torch.from_numpy(RL_train_label_zero[:split_ratio])


        totalInputPure = torch.cat((RL_td_rest,RL_train_data_zero_littlePure),dim=0)
        totalLabelPure = torch.cat((RL_tl_rest,RL_train_label_zero_littlePure),dim=0)

        totalInputSum = torch.cat((RL_td_rest, RL_train_data_zero_littleSum), dim=0)
        totalLabelSum = torch.cat((RL_tl_rest, RL_train_label_zero_littleSum), dim=0)

        theta_model_partPure = Prediction_lit_4REINFORCE1(save_dir='/home/a286/',
                                                              save_range=10,
                                                              beta4f1=100)

        dm4Pure = datamodule_4REINFORCE1(batch_size=1024,
                                    total_tdata=totalInputPure,
                                    total_tlabel=totalLabelPure,
                                    val_data=total_val_data,
                                    val_label=total_val_label)


        trainer_partPure = pl.Trainer(
                                  max_epochs=8,
                                  gpus=[0],
                                  strategy='dp',
                                  logger=False,
                                  enable_checkpointing=False,
                                  num_sanity_val_steps=0,
                                  enable_model_summary=None)


        trainer_partPure.fit(theta_model_partPure, dm4Pure)
        trainer_partPure.validate(theta_model_partPure,dm4Pure)

        REWARD_Pure =theta_model_partPure.avg_acc_lst_val_f1score[-1]

        resultPure.append(REWARD_Pure)
        fig = plt.figure(constrained_layout=True)
        ax1 = fig.add_subplot(2, 4, 1)
        ax1.plot(range(len(theta_model_partPure.avg_loss_lst_trn)), theta_model_partPure.avg_loss_lst_trn)
        ax1.set_title('train loss')
        ax2 = fig.add_subplot(2, 4, 2)
        ax2.plot(range(len(theta_model_partPure.avg_acc_lst_trn_PRECISION)), theta_model_partPure.avg_acc_lst_trn_PRECISION)
        ax2.set_title('train PRECISION')
        ax3 = fig.add_subplot(2, 4, 3)
        ax3.plot(range(len(theta_model_partPure.avg_acc_lst_trn_RECALL)), theta_model_partPure.avg_acc_lst_trn_RECALL)
        ax3.set_title('train RECALL')
        ax4 = fig.add_subplot(2, 4, 4)
        ax4.plot(range(len(theta_model_partPure.avg_acc_lst_trn_f1score)), theta_model_partPure.avg_acc_lst_trn_f1score)
        ax4.set_title('train F1 SCORE')

        ax5 = fig.add_subplot(2, 4, 5)
        ax5.plot(range(len(theta_model_partPure.avg_loss_lst_val)), theta_model_partPure.avg_loss_lst_val)
        ax5.set_title('val loss')
        ax6 = fig.add_subplot(2, 4, 6)
        ax6.plot(range(len(theta_model_partPure.avg_acc_lst_val_PRECISION)), theta_model_partPure.avg_acc_lst_val_PRECISION)
        ax6.set_title('val PRECISION')
        ax7 = fig.add_subplot(2, 4, 7)
        ax7.plot(range(len(theta_model_partPure.avg_acc_lst_val_RECALL)), theta_model_partPure.avg_acc_lst_val_RECALL)
        ax7.set_title('val RECALL')
        ax8 = fig.add_subplot(2, 4, 8)
        ax8.plot(range(len(theta_model_partPure.avg_acc_lst_val_f1score)), theta_model_partPure.avg_acc_lst_val_f1score)
        ax8.set_title('val F1 SCORE')

        plt.savefig(eachDir + 'inner_model_resultPure.png', dpi=200)
        print('saving plot complete!')
        plt.close()

        #############################################################################################
        #############################################################################################
        #############################################################################################
        #############################################################################################
        #############################################################################################
        #############################################################################################

        theta_model_partSum = Prediction_lit_4REINFORCE1(save_dir='/home/a286winteriscoming/',
                                                          save_range=10,
                                                          beta4f1=100)

        dm4Sum = datamodule_4REINFORCE1(batch_size=1024,
                                         total_tdata=totalInputSum,
                                         total_tlabel=totalLabelSum,
                                         val_data=total_val_data,
                                         val_label=total_val_label)

        time1 = time.time()
        avg_time = []
        trainer_partSum = pl.Trainer(
                                      max_epochs=8,
                                      gpus=[0],
                                      strategy='dp',
                                      logger=False,
                                      enable_checkpointing=False,
                                      num_sanity_val_steps=0,
                                      enable_model_summary=None)
        time2 = time.time()
        print('----------------------------------------------------------------------')

        trainer_partSum.fit(theta_model_partSum, dm4Sum)
        trainer_partSum.validate(theta_model_partSum, dm4Sum)

        REWARD_Sum = theta_model_partSum.avg_acc_lst_val_f1score[-1]

        resultSum.append(REWARD_Sum)

        fig = plt.figure(constrained_layout=True)
        ax1 = fig.add_subplot(2, 4, 1)
        ax1.plot(range(len(theta_model_partSum.avg_loss_lst_trn)), theta_model_partSum.avg_loss_lst_trn)
        ax1.set_title('train loss')
        ax2 = fig.add_subplot(2, 4, 2)
        ax2.plot(range(len(theta_model_partSum.avg_acc_lst_trn_PRECISION)), theta_model_partSum.avg_acc_lst_trn_PRECISION)
        ax2.set_title('train PRECISION')
        ax3 = fig.add_subplot(2, 4, 3)
        ax3.plot(range(len(theta_model_partSum.avg_acc_lst_trn_RECALL)), theta_model_partSum.avg_acc_lst_trn_RECALL)
        ax3.set_title('train RECALL')
        ax4 = fig.add_subplot(2, 4, 4)
        ax4.plot(range(len(theta_model_partSum.avg_acc_lst_trn_f1score)), theta_model_partSum.avg_acc_lst_trn_f1score)
        ax4.set_title('train F1 SCORE')

        ax5 = fig.add_subplot(2, 4, 5)
        ax5.plot(range(len(theta_model_partSum.avg_loss_lst_val)), theta_model_partSum.avg_loss_lst_val)
        ax5.set_title('val loss')
        ax6 = fig.add_subplot(2, 4, 6)
        ax6.plot(range(len(theta_model_partSum.avg_acc_lst_val_PRECISION)), theta_model_partSum.avg_acc_lst_val_PRECISION)
        ax6.set_title('val PRECISION')
        ax7 = fig.add_subplot(2, 4, 7)
        ax7.plot(range(len(theta_model_partSum.avg_acc_lst_val_RECALL)), theta_model_partSum.avg_acc_lst_val_RECALL)
        ax7.set_title('val RECALL')
        ax8 = fig.add_subplot(2, 4, 8)
        ax8.plot(range(len(theta_model_partSum.avg_acc_lst_val_f1score)), theta_model_partSum.avg_acc_lst_val_f1score)
        ax8.set_title('val F1 SCORE')

        plt.savefig(eachDir + 'inner_model_resultSum.png', dpi=200)
        print('saving plot complete!')
        plt.close()


    plt.plot(range(len(resultPure)),resultPure,'r')
    plt.plot(range(len(resultSum)),resultSum,'c')
    plt.xlabel('iteration')
    plt.ylabel('f1 beta score')
    plt.savefig(eachDir+'scoreDiff.png',dpi=200)
    plt.close()


