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


class simple_torch2(nn.Module):
    def __init__(self,
                 gamma,
                 eps,
                 rl_lr,
                 rl_b_size,
                 theta_b_size,
                 reward_normalize,
                 val_data,
                 val_label,
                 rwd_spread,
                 beta4f1,
                 inner_max_step,
                 theta_stop_threshold,
                 rl_stop_threshold,
                 test_fle_down_path,
                 theta_gpu_num,
                 model_save_load_path,
                 theta_max_epch,
                 max_ep,
                 conv_crit_num,
                 data_cut_num,
                 iter_to_accumul):
        super(simple_torch2, self).__init__()

        self.test_fle_down_path = test_fle_down_path
        self.model_save_load_path = model_save_load_path

        ####################################MODEL SETTINGG##############################3

        self.REINFORCE_model = ResNet4one(block=BasicBlock4one, num_blocks=[2, 2, 2, 2], num_classes=2, mnst_ver=True)

        ####################################MODEL SETTINGG##############################3

        ##########################VARS for RL model##################################
        self.loss_lst_trn = []
        self.loss_lst_trn_tmp = []
        self.loss_lst_val = []
        self.loss_lst_val_tmp = []

        self.policy_saved_log_probs_lst = []
        self.R_lst = []
        self.policy_saved_log_probs_lst_val = []
        self.R_lst_val = []

        self.total_reward_lst_trn = []
        self.total_reward_lst_trn_tmp = []
        self.total_reward_lst_val = []
        self.total_reward_lst_val_tmp = []

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
        self.iter_to_accumul = iter_to_accumul

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
        print('    flushing lst on pl level complete')

    def REINFORCE_LOSS(self, action_prob, reward,inputLen,totalLen):

        return action_prob * reward

    def step(self, action,inputLen,totalLen):

        inputs_data, inputs_label = action[0], action[1]


        num_one = torch.sum(inputs_label)
        num_zero = len(inputs_label) - num_one


        if num_zero+num_one != 0 and inputLen !=0:
            reward = -float((num_one-num_zero)/(num_one+num_zero)) * (inputLen/totalLen)
        else:
            reward = -1


        print(f'    num_one : {num_one} and num_zero : {num_zero} So, reward is : {reward}')
        done = True
        info = 'step complete'

        return reward, done, info

    def training_step(self, RL_td_zero,RL_tl_zero,RL_td_rest,RL_tl_rest):

        self.REINFORCE_model.train()



        with torch.set_grad_enabled(True):

            RL_td_rest = RL_td_rest[:self.data_cut_num]
            RL_tl_rest = RL_tl_rest[:self.data_cut_num]

            RL_td_zero = RL_td_zero[:self.data_cut_num]
            RL_tl_zero = RL_tl_zero[:self.data_cut_num]

            print(f'    shape of b_input which is rest only is : {RL_td_rest.shape}')
            print(f'    shape of b_input which is rest only is : {RL_td_rest.size()}')
            print(f'    shape of b_input which is zero only is : {RL_td_zero.shape}')
            print(f'    shape of label which is zero only is : {RL_tl_zero.shape}')

            train_data = TensorDataset(torch.cat((RL_td_rest,RL_td_zero),dim=0),torch.cat((RL_tl_rest, RL_tl_zero),dim=0) )
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.rl_b_size, num_workers=2)

            for idx,(b_input, b_label) in enumerate(train_dataloader):

                action_probs = self.forward(b_input)
                m = Categorical(action_probs)
                action = m.sample()
                action_bool = action.clone().detach().bool()
                print(f'    size of action_bool for {idx}/{len(train_dataloader)} is : {action_bool.size()}')

                self.policy_saved_log_probs_lst.append(torch.sum(m.log_prob(action)))

                good_data_tensor = b_input.clone().detach()
                good_label_tensor = b_label.clone().detach()
                data_filter_tensor = action_bool.clone().detach()


                if idx >= self.max_ep:
                    break

                print(f'    size of filterd data for {idx}/{len(train_dataloader)} is : {good_data_tensor[data_filter_tensor].size()}')
                print(f'    size of filterd label for {idx}/{len(train_dataloader)} is : {good_label_tensor[data_filter_tensor].size()}')

                action_4_step = [good_data_tensor[data_filter_tensor],
                                 good_label_tensor[data_filter_tensor]]

                reward, done, info = self.step(action=action_4_step,
                                               inputLen=len(action_4_step[1]),
                                               totalLen=len(action_4_step[1]))

                self.R_lst.append(reward)
                self.total_reward_lst_trn_tmp.append(reward+1) #TO shift reward from minus to 0<= reward<=1

                print(f'    self.R_lst for {idx}/{len(train_dataloader)} is : {self.R_lst}')

                Return = 0
                policy_loss = []
                Returns = []

                for r in self.R_lst[::-1]:
                    Return = r + self.gamma * Return
                    Returns.insert(0,Return)

                Returns = torch.tensor(Returns)


                for log_prob, R in zip(self.policy_saved_log_probs_lst, Returns):
                    policy_loss.append(-log_prob * R)

                print(f'    policy loss(list) is : {policy_loss}')
                print(f'    Returns(list) is : {Returns}')

                policy_loss = torch.sum(torch.stack(policy_loss))/self.iter_to_accumul
                # torch.sum(torch.cat(policy_loss))
                self.loss_lst_trn_tmp.append(float(policy_loss.item()))

                policy_loss.backward()

                if idx % self.iter_to_accumul == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    ################# mean of each append to lst for plot###########
                    self.loss_lst_trn.append(np.mean(self.loss_lst_trn_tmp))
                    self.total_reward_lst_trn.append(np.mean(self.total_reward_lst_trn_tmp))
                    ################# mean of each append to lst for plot###########
                    ###########flush#############
                    self.loss_lst_trn_tmp = []
                    self.total_reward_lst_trn_tmp = []
                    ###########flush#############
                print('    gradient optimization done')
                print('')


                self.flush_lst()
                print(f'policy loss is : {policy_loss}')


        torch.set_grad_enabled(False)
        self.REINFORCE_model.eval()

    def validation_step(self, RL_valInput,RL_valLabel):

        validationData = TensorDataset(RL_valInput,RL_valLabel)
        validationSampler = SequentialSampler(validationData)
        validationDataloader = DataLoader(validationData, sampler=validationSampler, batch_size=self.rl_b_size,
                                           num_workers=2)

        self.REINFORCE_model.eval()

        with torch.set_grad_enabled(False):

            for idx, (valBInput,valBLabel) in enumerate(validationDataloader):

                action_probs = self.forward(valBInput)
                m = Categorical(action_probs)
                action = m.sample()
                action_bool = action.clone().detach().bool()

                self.policy_saved_log_probs_lst_val.append(torch.sum(m.log_prob(action)))

                if idx == 0:
                    good_data_tensor = valBInput.clone().detach()
                    good_label_tensor = valBLabel.clone().detach()
                    data_filter_tensor = action_bool.clone().detach()
                else:

                    good_data_tensor = torch.cat((good_data_tensor,valBInput.clone().detach()),dim=0)
                    good_label_tensor = torch.cat((good_label_tensor,valBLabel.clone().detach()),dim=0)
                    data_filter_tensor = torch.cat((data_filter_tensor,action_bool.clone().detach()),dim=0)


            print(f'    size of filterd data is : {good_data_tensor[data_filter_tensor].size()}')
            print(f'    size of filterd label is : {good_label_tensor[data_filter_tensor].size()}')


            action_4_step = [good_data_tensor[data_filter_tensor],
                             good_label_tensor[data_filter_tensor]]

            reward, done, info = self.step(action=action_4_step,
                                           inputLen=len(action_4_step[1]),
                                           totalLen=len(action_4_step[1]))

            self.R_lst_val.append(reward)
            self.total_reward_lst_val.append(reward + 1)  # TO shift reward from minus to 0<= reward<=1

            Return = 0
            policy_loss = []
            Returns = []

            for r in self.R_lst_val[::-1]:
                Return = r + self.gamma * Return
                Returns.insert(0, Return)

            Returns = torch.tensor(Returns)

            for log_prob, R in zip(self.policy_saved_log_probs_lst_val, Returns):
                policy_loss.append(-log_prob * R)

            print(f'    policy loss(list) is : {policy_loss}')
            print(f'    Returns(list) is : {1 + Returns}')

            policy_loss = torch.sum(torch.stack(policy_loss))
            # torch.sum(torch.cat(policy_loss))
            self.loss_lst_val.append(float(policy_loss.item()))

        self.flush_lst()
        self.REINFORCE_model.train()

    def validationStepEnd(self):


        fig = plt.figure(constrained_layout=True)
        ax1 = fig.add_subplot(1, 4, 1)
        ax1.plot(range(len(self.loss_lst_trn)), self.loss_lst_trn)
        ax1.set_title('trn loss')
        ax2 = fig.add_subplot(1, 4, 2)
        ax2.plot(range(len(self.total_reward_lst_trn)), self.total_reward_lst_trn)
        ax2.set_title('trn reward')
        ax3 = fig.add_subplot(1, 4, 3)
        ax3.plot(range(len(self.loss_lst_val)), self.loss_lst_val)
        ax3.set_title('val loss')
        ax4 = fig.add_subplot(1, 4, 4)
        ax4.plot(range(len(self.total_reward_lst_val)), self.total_reward_lst_val)
        ax4.set_title('val reward')

        # print(f'self.test_fle_down_path is : {self.test_fle_down_path}testplot.png')
        plt.savefig(self.test_fle_down_path+'RL_reward_plot.png', dpi=200)
        print('saving plot complete!')
        plt.close()


    def STARTTRNANDVAL(self,
                       i,
                       RL_td_zero,
                       RL_tl_zero,
                       RL_td_rest,
                       RL_tl_rest,
                       RL_valInput,
                       RL_valLabel
                       ):

        print(f'start {i}th training')
        self.training_step(RL_td_zero,RL_tl_zero,RL_td_rest,RL_tl_rest)
        print(f'{i}th training complete')
        print(f'start {i}th validation')
        self.validation_step(RL_valInput=RL_valInput,RL_valLabel=RL_valLabel)
        self.validationStepEnd()
        print(f'{i}th validation complete')


