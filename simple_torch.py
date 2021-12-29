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
from MK_NOISED_DATA import mk_noisy_data
from save_funcs import load_my_model


class simple_torch(nn.Module):
    def __init__(self, gamma,eps,rl_lr,rl_b_size,theta_b_size,reward_normalize,val_data,val_label,rwd_spread,beta4f1,
                 inner_max_step,theta_stop_threshold,rl_stop_threshold,test_fle_down_path,theta_gpu_num,
                 model_save_load_path,theta_max_epch,max_ep,conv_crit_num,data_cut_num):
        super(simple_torch, self).__init__()

        self.test_fle_down_path = test_fle_down_path
        self.model_save_load_path = model_save_load_path

        ####################################MODEL SETTINGG##############################3

        self.REINFORCE_model = ResNet4one(block=BasicBlock4one, num_blocks=[2, 2, 2, 2], num_classes=2, mnst_ver=True)
        self.model_num_now = 0
        ####################################MODEL SETTINGG##############################3

        ##########################VARS for RL model##################################
        self.loss_lst_trn = []
        self.loss_lst_val = []

        self.policy_saved_log_probs_lst = []
        self.R_lst = []
        self.policy_saved_log_probs_lst_val = []
        self.R_lst_val = []

        self.total_reward_lst_trn = []

        self.automatic_optimization = False
        self.gamma = gamma
        self.eps = eps
        self.rl_lr = rl_lr
        self.rl_b_size = rl_b_size
        self.rl_stop_threshold = rl_stop_threshold
        self.reward_normalize = reward_normalize
        self.max_ep = max_ep
        self.beta4f1 = beta4f1
        self.data_cut_num = data_cut_num


        self.rwd_spread = rwd_spread

        self.optimizer = Adam(self.REINFORCE_model.parameters(),
                              lr=self.rl_lr,  # 학습률
                              eps=self.eps  # 0으로 나누는 것을 방지하기 위한 epsilon 값
                              )
        ##########################VARS for RL model##################################

        ##########################VARS for INNER THETA model##################################
        self.theta_b_size = theta_b_size
        self.theta_max_epch = theta_max_epch
        self.theta_stop_threshold = theta_stop_threshold
        self.theta_gpu_num = theta_gpu_num
        self.inner_max_step = inner_max_step
        self.conv_crit_num = conv_crit_num
        ##########################VARS for INNER THETA model##################################


        test_dataset = MNIST(self.test_fle_down_path, train=False, download=True)

    def forward(self, x):
        # print(f'RL part input is in device : {x.device}')
        probs_softmax = F.softmax(self.REINFORCE_model(x.float()),dim=1)
        # print(f'Rl part output is in device : {probs_softmax.device}')


        return probs_softmax

    def flush_lst(self):
        self.policy_saved_log_probs_lst = []
        self.policy_saved_log_probs_lst_val = []
        self.R_lst = []
        self.R_lst_val = []
        print('flushing lst on pl level complete')

    def REINFORCE_LOSS(self, action_prob, reward):

        return action_prob * reward

    def step(self, action,training_num):

        inputs_data, inputs_label = action[0], action[1]


        num_one = torch.sum(inputs_label)
        num_zero = len(inputs_label) - num_one


        if num_zero+num_one != 0:
            reward = float(num_zero/(num_one+num_zero))
        else:
            reward = -0.1

        print(f'num_one : {num_one} and num_zero : {num_zero} So, reward is : {reward}')
        done = True
        info = 'step complete'

        return reward, done, info

    def training_step(self, RL_td_zero,RL_tl_zero,RL_td_rest,RL_tl_rest,training_num):

        print(f'shape of b_input which is zero only is : {RL_td_rest.shape}')
        RL_td_rest = torch.from_numpy(RL_td_rest).unsqueeze(1)[:self.data_cut_num]
        print(f'shape of b_input which is zero only is : {RL_td_rest.size()}')
        RL_tl_rest = torch.ones_like(torch.from_numpy(RL_tl_rest))[:self.data_cut_num]

        RL_td_zero = RL_td_zero[:self.data_cut_num]
        RL_tl_zero = RL_tl_zero[:self.data_cut_num]

        print(f'shape of b_input which is zero only is : {RL_td_zero.shape}')
        print(f'shape of label which is zero only is : {RL_tl_zero.shape}')
        print(f'size of RL train_data rest is : {RL_td_rest.size()}')

        print('train_dataloading.......')
        train_data = TensorDataset(torch.cat((RL_td_rest,RL_td_zero),dim=0),torch.cat((RL_tl_rest, RL_tl_zero),dim=0) )
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.rl_b_size, num_workers=0)
        print('train_dataloading done....')

        print('train starttt!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        count_num = 0

        for b_input, b_label in train_dataloader:
            if count_num % 100 == 0:
                print(f'prop ing {count_num}th data')
            action_probs = self.forward(b_input)
            m = Categorical(action_probs)
            action = m.sample()
            action_bool = action.clone().detach().bool()

            self.policy_saved_log_probs_lst.append(m.log_prob(action))
            self.R_lst.append(0)

            #print(b_input.size())

            if count_num == 0:
                good_data_tensor = b_input.clone().detach()
                good_label_tensor = b_label.clone().detach()
                data_filter_tensor = action_bool.clone().detach()
            else:
                good_data_tensor = torch.cat((good_data_tensor,b_input.clone().detach()),dim=0)
                good_label_tensor = torch.cat((good_label_tensor,b_label.clone().detach()),dim=0)
                data_filter_tensor = torch.cat((data_filter_tensor,action_bool.clone().detach()),dim=0)
            #print(count_num, good_data_tensor.size(), good_label_tensor.size(), data_filter_tensor.size())
            #print(f'good_data_tensor is in device: {good_data_tensor.device}')

            count_num +=1

        print('rl prop complete')

        print(f'size of filterd data is : {good_data_tensor[data_filter_tensor].size()}')
        print(f'size of filterd label is : {good_label_tensor[data_filter_tensor].size()}')

        del train_data
        del train_sampler
        del train_dataloader


        action_4_step = [good_data_tensor[data_filter_tensor],
                         good_label_tensor[data_filter_tensor]]

        reward, done, info = self.step(action=action_4_step, training_num=training_num)


        if self.rwd_spread == True:
            self.R_lst = [reward/len(self.R_lst) for i in range(len(self.R_lst))]
        else:
            self.R_lst = self.R_lst[:-1]
            self.R_lst.append(reward)

        self.total_reward_lst_trn.append(reward)

        Return = 0
        policy_loss = []
        Returns = []

        for r in self.R_lst[::-1]:
            Return = r + self.gamma * Return
            Returns.insert(0,Return)
        Returns = torch.tensor(Returns)
        if self.reward_normalize == True:
            Returns = (Returns - Returns.mean()) / (Returns.std() + self.eps)

        for log_prob, R in zip(self.policy_saved_log_probs_lst, Returns):
            policy_loss.append(-log_prob * R)

        print(f'policy loss(list) is : {policy_loss}')
        print(f'Returns(list) is : {Returns}')
        policy_loss = torch.cat(policy_loss).sum()
        self.loss_lst_trn.append(float(policy_loss.item()))
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        print('gradient optimization done')

        #print(f'self.loss_lst_trn is : {self.loss_lst_trn}')
        #print(f'self.total_rwd_lst_trn is : {self.total_reward_lst_trn}')

        if training_num % 30 ==0:
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.plot(range(len(self.loss_lst_trn)), self.loss_lst_trn)
            ax1.set_title('loss')
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.plot(range(len(self.total_reward_lst_trn)), self.total_reward_lst_trn)
            ax2.set_title('reward')

            print(f'self.test_fle_down_path is : {self.test_fle_down_path}testplot.png')
            plt.savefig(self.test_fle_down_path+'RL_reward_plot.png', dpi=400)
            print('saving plot complete!')
            plt.close()


        good_data_lst = []
        good_label_lst = []
        data_filter_lst = []

        self.flush_lst()

        return policy_loss, reward

    def validation_step(self, validation_dataloader):
        print('val starttt!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        with torch.no_grad():
            for val_b_input, val_b_label in validation_dataloader:
                action_probs = self.forward(val_b_input)

                m = Categorical(action_probs)
                action = m.sample()
                action_bool = m.sample().clone().detach().bool()

                input_action4theta = [val_b_input[action_bool].clone().detach(), val_b_label[action_bool].clone().detach()]
                print(f'shape of input right before theta is : {input_action4theta[0].size()}')
                print(f'input right before theta is in device : {input_action4theta[0].device}')
                print(f'len of input_action4theta for validationRL is : {len(input_action4theta[0])}')

                reward, done, info = self.step(action=input_action4theta, stop_threshold=self.stop_threshold,
                                               weight4zero=self.weight4zero)

                reward = reward[0] + self.weight4zero*reward[1]

                loss = self.REINFORCE_LOSS(-1 * torch.mean(m.log_prob(action)), reward)

                break

        return loss
#######################TEST LINE ###################
#######################TEST LINE ###################
#######################TEST LINE ###################
#######################TEST LINE ###################
#######################TEST LINE ###################
