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


class REINFORCE_TORCH(nn.Module):
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
                 INNER_MAX_STEP,
                 reward_method,
                 WINDOW,
                 conv_crit_num):
        super(REINFORCE_TORCH, self).__init__()

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
        self.INNER_MAX_STEP =INNER_MAX_STEP
        self.reward_method = reward_method
        self.WINDOW = WINDOW
        self.pi4window = 0


        ########################do validation dataset change##########################
        self.val_label_zero = val_label[val_label == 0]
        self.val_label_rest = val_label[val_label != 0]

        self.val_data_zero = val_data[val_label == 0]
        self.val_data_rest = val_data[val_label != 0]
        self.val_label_rest = torch.ones_like(self.val_label_rest)

        self.val_data = torch.cat((self.val_data_rest,self.val_data_zero),dim=0)
        self.val_label = torch.cat((self.val_label_rest,self.val_label_zero),dim=0)

        del self.val_label_zero
        del self.val_label_rest
        del self.val_data_rest
        del self.val_data_zero
        ######################################do validation dataset change#################
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

        inputs_data4Theta, inputs_label4Theta = action[0], action[1]

        theta_model_part = Prediction_lit_4REINFORCE1(save_dir=self.test_fle_down_path,
                                                      save_range=10,
                                                      beta4f1=self.beta4f1)

        dm4Theta = datamodule_4REINFORCE1(batch_size=self.theta_b_size,
                                    total_tdata=inputs_data4Theta,
                                    total_tlabel=inputs_label4Theta,
                                    val_data=self.val_data,
                                    val_label=self.val_label)



        time1 = time.time()
        avg_time = []
        trainer_part = pl.Trainer(max_steps=self.INNER_MAX_STEP,
                                  max_epochs=1,
                                  gpus=self.theta_gpu_num,
                                  strategy='dp',
                                  logger=False,
                                  enable_checkpointing=False,
                                  num_sanity_val_steps=0,
                                  enable_model_summary=None)
        time2 = time.time()
        print('----------------------------------------------------------------------')

        trainer_part.fit(theta_model_part, dm4Theta)
        trainer_part.validate(theta_model_part,dm4Theta)
        del trainer_part
        theta_model_part.flush_lst()

        fig = plt.figure()
        ax1 = fig.add_subplot(2, 4, 1)
        ax1.plot(range(len(theta_model_part.avg_loss_lst_trn)), theta_model_part.avg_loss_lst_trn)
        ax1.set_title('train loss')
        ax2 = fig.add_subplot(2, 4, 2)
        ax2.plot(range(len(theta_model_part.avg_acc_lst_trn_PRECISION)), theta_model_part.avg_acc_lst_trn_PRECISION)
        ax2.set_title('train PRECISION')
        ax3 = fig.add_subplot(2, 4, 3)
        ax3.plot(range(len(theta_model_part.avg_acc_lst_trn_RECALL)), theta_model_part.avg_acc_lst_trn_RECALL)
        ax3.set_title('train RECALL')
        ax4 = fig.add_subplot(2, 4, 4)
        ax4.plot(range(len(theta_model_part.avg_acc_lst_trn_f1score)), theta_model_part.avg_acc_lst_trn_f1score)
        ax4.set_title('train F1 SCORE')

        ax5 = fig.add_subplot(2, 4, 5)
        ax5.plot(range(len(theta_model_part.avg_loss_lst_val)), theta_model_part.avg_loss_lst_val)
        ax5.set_title('val loss')
        ax6 = fig.add_subplot(2, 4, 6)
        ax6.plot(range(len(theta_model_part.avg_acc_lst_val_PRECISION)), theta_model_part.avg_acc_lst_val_PRECISION)
        ax6.set_title('val PRECISION')
        ax7 = fig.add_subplot(2, 4, 7)
        ax7.plot(range(len(theta_model_part.avg_acc_lst_val_RECALL)), theta_model_part.avg_acc_lst_val_RECALL)
        ax7.set_title('val RECALL')
        ax8 = fig.add_subplot(2, 4, 8)
        ax8.plot(range(len(theta_model_part.avg_acc_lst_val_f1score)), theta_model_part.avg_acc_lst_val_f1score)
        ax8.set_title('val F1 SCORE')

        plt.savefig(self.test_fle_down_path + 'inner_model_result.png', dpi=200)
        print('saving plot complete!')
        plt.close()

        print('----------------------------------------------------------------------')
        print('      ')
        print('      ')
        print('      ')

        if self.reward_method == 'mean':
            reward = np.mean(theta_model_part.avg_acc_lst_val_f1score[-self.conv_crit_num:])
        if self.reward_method == 'last':
            reward = theta_model_part.avg_acc_lst_val_f1score[-1:][0]

        done = True
        info = 'step complete'

        theta_model_part.flush_lst()
        del theta_model_part
        #del trainer_part
        del dm4Theta

        return reward, done, info

    def training_step(self, RL_td_zero,RL_tl_zero,RL_td_rest,RL_tl_rest,training_num):

        self.REINFORCE_model.train()

        print(f'shape of b_input which is rest only is : {RL_td_rest.shape}')
        RL_td_rest = torch.from_numpy(RL_td_rest).unsqueeze(1)
        print(f'shape of b_input which is rest only is : {RL_td_rest.size()}')
        RL_tl_rest = torch.ones_like(torch.from_numpy(RL_tl_rest))

        print(f'shape of b_input which is zero only is : {RL_td_zero.shape}')
        print(f'shape of label which is zero only is : {RL_tl_zero.shape}')

        input_data4ori = torch.cat((RL_td_rest,RL_td_zero), dim=0)
        input_label4ori = torch.cat((RL_tl_rest,RL_tl_zero), dim=0)

        ori_model_part = Prediction_lit_4REINFORCE1(save_dir=self.test_fle_down_path,
                                                    save_range=10,
                                                    beta4f1=self.beta4f1)

        dm4Ori = datamodule_4REINFORCE1(batch_size=self.theta_b_size,
                                        total_tdata=input_data4ori,
                                        total_tlabel=input_label4ori,
                                        val_data=self.val_data,
                                        val_label=self.val_label)

        trainer_part4Ori = pl.Trainer(max_steps=self.INNER_MAX_STEP,
                                  max_epochs=1,
                                  gpus=self.theta_gpu_num,
                                  strategy='dp',
                                  logger=False,
                                  enable_checkpointing=False,
                                  num_sanity_val_steps=0,
                                  enable_model_summary=None)

        trainer_part4Ori.fit(ori_model_part,dm4Ori)
        trainer_part4Ori.validate(ori_model_part, dm4Ori)

        REWARD_ORI = ori_model_part.avg_acc_lst_val_f1score[-1]

        print('train_dataloading.......')
        train_data = TensorDataset(RL_td_zero, RL_tl_zero)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.rl_b_size, num_workers=2)
        print('train_dataloading done....')

        print('train starttt!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        with torch.set_grad_enabled(True):

            for b_input, b_label in train_dataloader:
                print(f'size of rl_td_rest is : {RL_td_rest.shape}')
                print(f'size of rl_td_zero is : {RL_td_zero.shape}')
                print(f'size of b_input 20222 is : {b_input.size()}')
                action_probs = self.forward(b_input)
                m = Categorical(action_probs)
                action = m.sample()
                action_bool = action.clone().detach().bool()

                self.policy_saved_log_probs_lst.append(m.log_prob(action))


                #print(b_input.size())


                good_data_tensor = b_input.clone().detach()
                good_label_tensor = b_label.clone().detach()
                data_filter_tensor = action_bool.clone().detach()
                # else:
                #     good_data_tensor = torch.cat((good_data_tensor,b_input.clone().detach()),dim=0)
                #     good_label_tensor = torch.cat((good_label_tensor,b_label.clone().detach()),dim=0)
                #     data_filter_tensor = torch.cat((data_filter_tensor,action_bool.clone().detach()),dim=0)


                print(f'size of filterd data is : {good_data_tensor[data_filter_tensor].size()}')
                print(f'size of filterd label is : {good_label_tensor[data_filter_tensor].size()}')


                action_4_step = [torch.cat((RL_td_rest,good_data_tensor[data_filter_tensor]),dim=0)
                ,torch.cat((RL_tl_rest,good_label_tensor[data_filter_tensor]),dim=0)]

                reward, done, info = self.step(action=action_4_step, training_num=training_num)

                rewardDiff = reward - self.pi4window
                self.pi4window = self.pi4window*((self.WINDOW-1)/self.WINDOW) + REWARD_ORI/self.WINDOW

                self.R_lst.append(reward)
                self.total_reward_lst_trn.append(reward)

                Return = 0
                policy_loss = []
                Returns = []

                print(f'self.R_lst is : {self.R_lst}')

                for r in self.R_lst[::-1]:
                    Return = r + self.gamma * Return
                    Returns.insert(0,Return)
                Returns = torch.tensor(Returns)
                print(f'Returns : {Returns}')
                # if self.reward_normalize == True:
                #     Returns = (Returns - Returns.mean()) / (Returns.std() + self.eps)

                for log_prob in self.policy_saved_log_probs_lst:
                    policy_loss.append(-log_prob * Returns)

                policy_loss = torch.sum(torch.cat(policy_loss))
                self.loss_lst_trn.append(float(policy_loss.item()))
                self.optimizer.zero_grad()
                policy_loss.backward()
                self.optimizer.step()
                print('gradient optimization done')

                # print(f'self.loss_lst_trn is : {self.loss_lst_trn}')
                # print(f'self.total_rwd_lst_trn is : {self.total_reward_lst_trn}')

                fig = plt.figure()
                ax1 = fig.add_subplot(1, 2, 1)
                ax1.plot(range(len(self.loss_lst_trn)), self.loss_lst_trn)
                ax1.set_title('loss')
                ax2 = fig.add_subplot(1, 2, 2)
                ax2.plot(range(len(self.total_reward_lst_trn)), self.total_reward_lst_trn)
                ax2.set_title('reward')


                # print(f'self.test_fle_down_path is : {self.test_fle_down_path}testplot.png')
                plt.savefig(self.test_fle_down_path+'RL_reward_plot.png', dpi=200)
                print('saving plot complete!')
                plt.close()

                self.flush_lst()
            torch.set_grad_enabled(False)
            self.REINFORCE_model.eval()
        return policy_loss, reward

    def validation_step(self, RL_td_zero,RL_tl_zero,RL_td_rest,RL_tl_rest,valNum):

        RL_td_rest = torch.from_numpy(RL_td_rest).unsqueeze(1)
        RL_tl_rest = torch.ones_like(torch.from_numpy(RL_tl_rest))


        LowLst = [i for i in range(5)]

        criterionResultLst= []
        dvrlResultLst = []

        print('val starttt!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.REINFORCE_model.eval()

        data4Criterion = torch.cat((RL_td_rest,RL_td_zero),dim=0)
        label4Criterion = torch.cat((RL_tl_rest,RL_tl_zero),dim=0)


        val_model_part_BASE = Prediction_lit_4REINFORCE1(save_dir=self.test_fle_down_path,
                                                    save_range=10,
                                                    beta4f1=self.beta4f1)

        dm4BASE = datamodule_4REINFORCE1(batch_size=self.theta_b_size,
                                        total_tdata=data4Criterion,
                                        total_tlabel=label4Criterion,
                                        val_data=self.val_data,
                                        val_label=self.val_label)

        trainer_part4BASE = pl.Trainer(max_steps=self.INNER_MAX_STEP,
                                  max_epochs=1,
                                  gpus=self.theta_gpu_num,
                                  strategy='dp',
                                  logger=False,
                                  enable_checkpointing=False,
                                  num_sanity_val_steps=0,
                                  enable_model_summary=None)

        trainer_part4BASE.fit(val_model_part_BASE,dm4BASE)
        trainer_part4BASE.validate(val_model_part_BASE,dm4BASE)

        rewardCriterion = val_model_part_BASE.avg_acc_lst_val_f1score[-1]

        del val_model_part_BASE
        del dm4BASE
        del trainer_part4BASE
        del data4Criterion
        del label4Criterion


        print('----------------------------------------')
        print('----------------------------------------')
        print('----------------------------------------')
        print('----------------------------------------')
        print('----------------------------------------')
        print('----------------------------------------')
        print('----------------------------------------')
        print('----------------------------------------')
        print('----------------------------------------')
        print('----------------------------------------')
        print('----------------------------------------')
        print('----------------------------------------')
        print('----------------------------------------')
        print('----------------------------------------')
        print('----------------------------------------')
        print('----------------------------------------')
        print('----------------------------------------')
        print('----------------------------------------')
        print('----------------------------------------')
        print('----------------------------------------')
        print('----------------------------------------')

        val_data = TensorDataset(RL_td_zero,RL_tl_zero)
        val_dataloader = DataLoader(val_data, shuffle=False, batch_size=self.rl_b_size, num_workers=4)



        with torch.set_grad_enabled(False):

            for idx,(inputVal,_) in enumerate(val_dataloader):

                dataValueBatch = self.forward(inputVal)[:,1]

                print(f'size of dataValueBatch.size() is : {dataValueBatch.size()}')


                if idx == 0:
                    datavalueTensor = dataValueBatch.clone().detach()
                else:
                    datavalueTensor = torch.cat((datavalueTensor,dataValueBatch.clone().detach()),dim=0)



        print(f'datavalueTensor.size() is : {datavalueTensor.size()}')



        for Low in LowLst:

            # lowRemovedTensor = torch.ge(datavalueTensor,Low)
            lowRemovedTensor = torch.bernoulli(datavalueTensor) > 0

            print(f'min of datavalue is : {torch.min(datavalueTensor)}')

            print(f'lowRemovedTensor is : {lowRemovedTensor}')
            print(f'size of lowRemovedTensor is : {lowRemovedTensor.size()}')
            newTdZero = RL_td_zero[lowRemovedTensor]
            newTlZero = RL_tl_zero[lowRemovedTensor]

            print(f'size of newTdZero is : {newTdZero.size()}')
            print(f'size of newTlZero is : {newTlZero.size()}')

            data4DVRL = torch.cat((RL_td_rest,newTdZero),dim=0)
            label4DVRL = torch.cat((RL_tl_rest,newTlZero),dim=0)

            val_model_part_DVRL = Prediction_lit_4REINFORCE1(save_dir=self.test_fle_down_path,
                                                        save_range=10,
                                                        beta4f1=self.beta4f1)

            dm4DVRL = datamodule_4REINFORCE1(batch_size=self.theta_b_size,
                                            total_tdata=data4DVRL,
                                            total_tlabel=label4DVRL,
                                            val_data=self.val_data,
                                            val_label=self.val_label)

            trainer_part4DVRL = pl.Trainer(max_steps=self.INNER_MAX_STEP,
                                      max_epochs=1,
                                      gpus=self.theta_gpu_num,
                                      strategy='dp',
                                      logger=False,
                                      enable_checkpointing=False,
                                      num_sanity_val_steps=0,
                                      enable_model_summary=None)

            trainer_part4DVRL.fit(val_model_part_DVRL, dm4DVRL)
            trainer_part4DVRL.validate(val_model_part_DVRL, dm4DVRL)

            rewardDVRL = val_model_part_DVRL.avg_acc_lst_val_f1score[-1]

            criterionResultLst.append(rewardCriterion)
            dvrlResultLst.append(rewardDVRL)

        plt.plot(LowLst,dvrlResultLst,'r')
        plt.plot(LowLst, criterionResultLst,'b')
        plt.xlabel('iteration')
        plt.savefig(self.test_fle_down_path+'rewardDiffwithLow'+'.png',dpi=200)
        print(f'saving plot for innnerStep : {self.INNER_MAX_STEP} for {valNum} done')
        plt.close()


        del data4DVRL
        del label4DVRL
        del val_model_part_DVRL
        del dm4DVRL
        del trainer_part4DVRL

        return
