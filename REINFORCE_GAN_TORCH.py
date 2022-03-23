import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
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
from MY_MODELS import ganGenerator1, ganDiscriminator1,ResNet4one,BasicBlock4one
from save_funcs import mk_name,lst2csv,createDirectory
from REINFORCE_DATAMODULES import datamodule_4REINFORCE1
from REINFORCE_INNER_MODELS import Prediction_lit_4REINFORCE1
from MK_NOISED_DATA import mk_noisy_data
from save_funcs import load_my_model


class REINFORCE_GAN_TORCH(nn.Module):
    def __init__(self,
                 gamma,
                 eps,
                 dNoise,
                 dHidden,
                 rl_lr,
                 rl_b_size,
                 gan_trn_bSize,
                 gan_val_bSize,
                 reward_normalize,
                 val_data,
                 val_label,
                 rwd_spread,
                 beta4f1,
                 theta_gpu_num,
                 theta_b_size,
                 rl_stop_threshold,
                 test_fle_down_path,
                 model_save_load_path,
                 model_save_load_pathG,
                 model_save_load_pathGbase,
                 model_save_load_pathD,
                 model_save_load_pathDVRL,
                 GLoadNum,
                 GbaseLoadNum,
                 DLoadNum,
                 DvrlLoadNum,
                 INNER_MAX_STEP,
                 max_step_trn,
                 max_step_val,
                 whichGanLoss,
                 reward_method,
                 val_num2genLst,
                 Num2Gen,
                 useDiff,
                 lsganA= 0,
                 lsganB = 1,
                 lsganC = 1,
                 ):
        super(REINFORCE_GAN_TORCH, self).__init__()

        self.test_fle_down_path = test_fle_down_path
        self.model_save_load_path = model_save_load_path
        self.model_save_load_pathG = model_save_load_pathG
        self.model_save_load_pathGbase = model_save_load_pathGbase
        self.model_save_load_pathD = model_save_load_pathD
        self.model_save_load_pathDVRL =model_save_load_pathDVRL

        self.GLoadNum = GLoadNum
        print('self.GLoadNum is :',self.GLoadNum)
        self.GbaseLoadNum = GbaseLoadNum
        self.DLoadNum = DLoadNum
        self.DvrlLoadNum = DvrlLoadNum

        self.dNoise = dNoise
        self.dHidden = dHidden
        self.whichGanLoss = whichGanLoss
        self.lsganA = lsganA
        self.lsganB =lsganB
        self.lsganC =lsganC
        self.theta_b_size = theta_b_size
        self.INNER_MAX_STEP = INNER_MAX_STEP

        ####################################MODEL SETTINGG##############################3

        self.REINFORCE_GAN_G = ganGenerator1(dNoise=self.dNoise,dHidden=self.dHidden)
        self.REINFORCE_GAN_GBASE = copy.deepcopy(self.REINFORCE_GAN_G)
        self.REINFORCE_GAN_D = ganDiscriminator1(dHidden=self.dHidden)
        self.REINFORCE_GAN_DBASE = copy.deepcopy(self.REINFORCE_GAN_D)
        self.REINFORCE_DVRL = ResNet4one(block=BasicBlock4one, num_blocks=[2, 2, 2, 2], num_classes=2, mnst_ver=True)
        self.model_num_now = 0
        ####################################MODEL SETTINGG##############################3

        ##########################VARS for RL model##################################
        self.loss_lst_trn = []
        self.loss_lst_val = []

        self.policy_saved_log_probs_lst = []
        self.R_lst = []
        self.policy_saved_log_probs_lst_val = []
        self.R_lst_val = []
        self.theta_gpu_num = theta_gpu_num
        self.total_reward_lst_trn = []

        self.automatic_optimization = False
        self.gamma = gamma
        self.eps = eps
        self.rl_lr = rl_lr
        self.rl_b_size = rl_b_size
        self.rl_stop_threshold = rl_stop_threshold
        self.reward_normalize = reward_normalize
        self.max_step_trn = max_step_trn
        self.max_step_val = max_step_val
        self.beta4f1 = beta4f1

        self.reward_method = reward_method

        self.pi4window = 0

        self.val_num2genLst = val_num2genLst
        self.Num2Gen = Num2Gen

        self.gan_trn_bSize = gan_trn_bSize
        self.gan_val_bSize = gan_val_bSize


        self.lossLstLossG_GAN = []
        self.lossLstLossG_DVRL = []
        self.lossLstLossD = []
        self.lossLstLossGBASE = []
        self.lossLstLossDBASE = []

        self.ProbFakeLst = []
        self.ProbRealLst = []
        self.ProbFakeLstBASE = []
        self.ProbRealLstBASE = []

        self.reward4plotGBaseLst =[]
        self.reward4plotGLst = []

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
        self.useDiff = useDiff

        self.optimizerD = Adam(self.REINFORCE_GAN_D.parameters(),
                              lr=self.rl_lr,  # 학습률
                              eps=self.eps  # 0으로 나누는 것을 방지하기 위한 epsilon 값
                              )

        self.optimizerG = Adam(self.REINFORCE_GAN_G.parameters(),
                               lr=self.rl_lr,  # 학습률
                               eps=self.eps  # 0으로 나누는 것을 방지하기 위한 epsilon 값
                               )
        self.optimizerGBASE = Adam(self.REINFORCE_GAN_GBASE.parameters(),
                               lr=self.rl_lr,  # 학습률
                               eps=self.eps  # 0으로 나누는 것을 방지하기 위한 epsilon 값
                               )
        self.optimizerDBASE = Adam(self.REINFORCE_GAN_DBASE.parameters(),
                               lr=self.rl_lr,  # 학습률
                               eps=self.eps  # 0으로 나누는 것을 방지하기 위한 epsilon 값
                               )
        self.optimizerDVRL = Adam(self.REINFORCE_DVRL.parameters(),
                               lr=self.rl_lr,  # 학습률
                               eps=self.eps  # 0으로 나누는 것을 방지하기 위한 epsilon 값
                               )

        ##########################VARS for RL model##################################


        test_dataset = MNIST(self.test_fle_down_path, train=False, download=True)

    def forwardGAN(self, x):

        GeneratedImages = self.REINFORCE_GAN_G(x)

        return GeneratedImages

    def forward(self,x):
        # print(f'RL part input is in device : {x.device}')
        probs_softmax = F.softmax(self.REINFORCE_DVRL(x.float()),dim=1)
        # print(f'Rl part output is in device : {probs_softmax.device}')


        return probs_softmax


    def loadSavedModel(self):
        try:

            self.REINFORCE_GAN_G.load_state_dict(torch.load(self.model_save_load_pathG+str(self.GLoadNum)+'.pt'))
            print(f'model GAN G loading {str(self.GLoadNum)} success')
        except:
            print(f'model GAN G loading failed... start with fresh model')
        try:

            self.REINFORCE_GAN_GBASE.load_state_dict(torch.load(self.model_save_load_pathGbase+str(self.GbaseLoadNum)+'.pt'))
            print(f'model GAN BASE loading {self.GbaseLoadNum} success')
        except:
            print(f'model GAN G base loading failed... start with fresh model')
        try:

            self.REINFORCE_GAN_D.load_state_dict(torch.load(self.model_save_load_pathD+str(self.DLoadNum)+'.pt'))
            print(f'model D loading {self.DLoadNum} success')
        except:
            print(f'model GAN D loading failed... start with fresh model')
        try:
            self.REINFORCE_DVRL.load_state_dict(torch.load(self.model_save_load_pathDVRL+str(self.DvrlLoadNum)+'.pt'))
            print(f'model DVRL loading {self.DvrlLoadNum} success')
        except:
            print(f'model DVRL loading failed... start with fresh model')

    def updateVars(self,**vars):

        for var,value in vars.items():
            if hasattr(self,var):
                print(f'{var} exist!!')
                print(f'value of {var} before update is {getattr(self, var)}')
                setattr(self,var,value)
                print(f'value of {var} after update is {getattr(self, var)}')
            else:
                print(f'{var} doesnt exist !!')

    def get_gaussianNoise_z(self,BSize):

        return torch.randn(BSize,self.dNoise)

    def flush_lst(self):
        self.policy_saved_log_probs_lst = []
        self.policy_saved_log_probs_lst_val = []
        self.R_lst = []
        self.R_lst_val = []
        print('flushing lst on pl level complete')

    def imshow(self,img,saveDir):
        img = (img + 1) / 2
        img = img.squeeze()
        np_img = img.numpy()
        plt.imshow(np_img, cmap='gray')
        # plt.show()
        plt.savefig(saveDir + 'GENERATED_IMAGE.png', dpi=200)
        plt.close()

    def imshow_grid(self,img,saveDir,showNum):
        img = utils.make_grid(img.cpu().detach())
        img = (img + 1) / 2
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

        randNum = random.random()
        if randNum < 0.1 and showNum % 10 ==0:
            plt.show()
        plt.savefig(saveDir + 'GENERATED_IMAGE_GRID.png', dpi=200)
        plt.close()

    def REINFORCE_LOSS(self, action_prob, reward):

        return action_prob * reward

    def GAN_LOSS(self,GorD,p_fake,p_real=None,):
        
        if self.whichGanLoss == 'vanilla':

            if GorD == 'D':
                lossFake = -1 * torch.log(1.0 - p_fake)
                lossReal = -1 * torch.log(p_real)
    
                lossD = (lossReal+lossFake).mean()
    
                return lossD
            if GorD == 'G':
                lossG = -1 * torch.log(p_fake).mean()
    
                return lossG
            
        if self.whichGanLoss == 'lsgan':

            if GorD == 'D':
                lossFake = 0.5 * torch.mean(torch.square(p_fake-self.lsganA))
                lossReal = 0.5 * torch.mean(torch.square(p_real-self.lsganB))

                lossD = (lossReal + lossFake).mean()

                return lossD
            if GorD == 'G':
                lossG = 0.5 * torch.mean(torch.square(p_fake-self.lsganC))

                return lossG

    def step(self, action):

        inputs_data4Theta, inputs_label4Theta = action[0], action[1]
        inputs_label4Theta = inputs_label4Theta.long()

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
        print('-----------------------------------------------------------------------------------')

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
        print('saving inner plot complete!')
        plt.close()


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
        print('----------------------------------------------------------------------------------------')
        print('')

        return reward, done, info


    def training_step(self, td2gen, tl2gen,RL_td_rest,RL_tl_rest):

        self.REINFORCE_GAN_G.train()
        self.REINFORCE_GAN_D.train()
        self.REINFORCE_GAN_GBASE.train()
        self.REINFORCE_GAN_DBASE.train()
        self.REINFORCE_DVRL.train()

        RL_tl_rest = torch.ones_like(RL_tl_rest)

        ########################################################################
        ########################################################################
        ########################################################################
        ########################################################################
        #################   code for baseline G training########################

        print('training G Base Start..... training G Base Start..... training G Base Start..... training G Base Start.....')
        print(
            'training G Base Start..... training G Base Start..... training G Base Start..... training G Base Start.....')
        print(
            'training G Base Start..... training G Base Start..... training G Base Start..... training G Base Start.....')
        print(' ')

        with torch.set_grad_enabled(True):

            train_data = TensorDataset(td2gen, tl2gen)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data,
                                          sampler=train_sampler,
                                          batch_size=self.gan_trn_bSize,
                                          num_workers=1)

            print(f'size of original data is : {td2gen.size()}')

            for idx,(bInput, bLabel) in enumerate(train_dataloader):
                print('.'*(idx+1))

                bInput = bInput.float()

                self.optimizerGBASE.zero_grad()
                self.optimizerDBASE.zero_grad()

                noiseZ = self.get_gaussianNoise_z(bInput.size(0))

                p_real = self.REINFORCE_GAN_DBASE(bInput)
                p_fake = self.REINFORCE_GAN_DBASE(self.REINFORCE_GAN_GBASE(noiseZ))

                lossD = self.GAN_LOSS(p_fake=p_fake,p_real=p_real,GorD='D')
                lossD.backward()
                self.optimizerDBASE.step()

                noiseZ = self.get_gaussianNoise_z(bInput.size(0))
                p_fake = self.REINFORCE_GAN_DBASE(self.REINFORCE_GAN_GBASE(noiseZ))

                self.optimizerGBASE.zero_grad()
                lossG = self.GAN_LOSS(p_fake=p_fake,GorD='G')
                lossG.backward()
                self.optimizerGBASE.step()

                self.lossLstLossGBASE.append(lossG.item())
                self.lossLstLossDBASE.append(lossD.item())
                self.ProbFakeLstBASE.append(torch.mean(p_fake).item())
                self.ProbRealLstBASE.append(torch.mean(p_real).item())
        print('training G Base complete!! training G Base complete!! training G Base complete!! training G Base complete!! ')
        print('training G Base complete!! training G Base complete!! training G Base complete!! training G Base complete!! ')
        print('training G Base complete!! training G Base complete!! training G Base complete!! training G Base complete!! ')
        print('')
        print('')


        #################   code for baseline G training########################
        ########################################################################
        ########################################################################
        ########################################################################
        ########################################################################







        ########################################################################
        ########################################################################
        ########################################################################
        ########################################################################
        #################   code for training GAN   ############################
        print('training real G start.... training real G start.... training real G start.... training real G start.... ')
        print(
            'training real G start.... training real G start.... training real G start.... training real G start.... ')
        print(
            'training real G start.... training real G start.... training real G start.... training real G start.... ')

        torch.set_grad_enabled(True)
        train_data = TensorDataset(td2gen, tl2gen)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data,
                                      sampler=train_sampler,
                                      batch_size=self.gan_trn_bSize,
                                      num_workers=1)

        print(f'size of original data is : {td2gen.size()}')

        for idx,(bInput, bLabel) in enumerate(train_dataloader):
            print('.'*(idx+1))

            bInput = bInput.float()

            self.optimizerG.zero_grad()
            self.optimizerD.zero_grad()

            noiseZ = self.get_gaussianNoise_z(bInput.size(0))

            p_real = self.REINFORCE_GAN_D(bInput)
            p_fake = self.REINFORCE_GAN_D(self.REINFORCE_GAN_G(noiseZ))

            lossD = self.GAN_LOSS(p_fake=p_fake,p_real=p_real,GorD='D')
            lossD.backward()
            self.optimizerD.step()
            self.optimizerG.zero_grad()

            noiseZ = self.get_gaussianNoise_z(bInput.size(0))
            GeneratedImgbyG = self.REINFORCE_GAN_G(noiseZ)
            # for normal GAN Loss
            p_fake = self.REINFORCE_GAN_D(GeneratedImgbyG)
            # for DVRL loss
            for param in self.REINFORCE_DVRL.parameters():
                param.requires_grad = False
            selectionProb = self.REINFORCE_DVRL(GeneratedImgbyG)[:,1]

            lossG_GAN = self.GAN_LOSS(p_fake=p_fake,GorD='G')
            lossG_DVRL = torch.mean(torch.square(selectionProb-1.0))
            totalLossG = lossG_GAN + lossG_DVRL
            totalLossG.backward()

            self.lossLstLossG_GAN.append(lossG_GAN.item())
            self.lossLstLossG_DVRL.append(lossG_DVRL.item())
            self.lossLstLossD.append(lossD.item())
            self.ProbFakeLst.append(torch.mean(p_fake).item())
            self.ProbRealLst.append(torch.mean(p_real).item())
            self.optimizerG.step()
        print('training real G complete!!! training real G complete!!! training real G complete!!! training real G complete!!! ')
        print(
            'training real G complete!!! training real G complete!!! training real G complete!!! training real G complete!!! ')
        print(
            'training real G complete!!! training real G complete!!! training real G complete!!! training real G complete!!! ')
        print('')
        print('')

        #################   code for training GAN   ############################
        ########################################################################
        ########################################################################
        ########################################################################
        ########################################################################




        ########################################################################
        ########################################################################
        ########################################################################
        ########################################################################
        #################   code for baseline reward for DVRL ##################

        if self.useDiff == True:
            print('training DVRL BASE start....')
            print(' ')
            print(' ')
            print(' ')
            print(' ')
            with torch.set_grad_enabled(False):


                RL_tl_rest = torch.ones_like(torch.from_numpy(RL_tl_rest))

                input_data4ori = torch.cat((RL_td_rest, td2gen), dim=0)
                input_label4ori = torch.cat((RL_tl_rest, tl2gen), dim=0)

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

                trainer_part4Ori.fit(ori_model_part, dm4Ori)
                trainer_part4Ori.validate(ori_model_part, dm4Ori)

                REWARD_ORI = ori_model_part.avg_acc_lst_val_f1score[-1]
                print('training real G complete!!!')
                print(' ')
                print(' ')
                print(' ')
                print(' ')

        #################   code for baseline reward for DVRL ##################
        ########################################################################
        ########################################################################
        ########################################################################
        ########################################################################



        ########################################################################
        ########################################################################
        ########################################################################
        ########################################################################
        #################   code for training DVRL  ############################


        with torch.set_grad_enabled(False):

            for i in range(self.Num2Gen):

                noiseZ4Gen = self.get_gaussianNoise_z(self.gan_trn_bSize)
                GeneratedImg = self.REINFORCE_GAN_G(noiseZ4Gen)

                if i ==0:
                    RL_td_zero = td2gen
                else:
                    RL_td_zero = torch.cat((RL_td_zero,GeneratedImg),dim=0)


        print('DVRL train_dataloading.......')
        DVRL_train_data = TensorDataset(RL_td_zero, torch.zeros(RL_td_zero.size(0)))
        DVRL_train_sampler = RandomSampler(train_data)
        DVRL_train_dataloader = DataLoader(DVRL_train_data,
                                           sampler=DVRL_train_sampler,
                                           batch_size=self.rl_b_size,
                                           num_workers=0)
        print('DVRL train_dataloading complete!')
        print(f'size of input for DVRL is : {RL_td_zero.size()} ')


        print('DVRL train starttt!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        for param in self.REINFORCE_DVRL.parameters():
            param.requires_grad = True

        with torch.set_grad_enabled(True):

            for idx,(b_input, b_label) in enumerate(DVRL_train_dataloader):


                print(f'size of b_input for {idx} th step of DVRL is : {b_input.size()}')

                action_probs = self.forward(b_input)
                m = Categorical(action_probs)
                action = m.sample()
                action_bool = action.clone().detach().bool()

                self.policy_saved_log_probs_lst.append(m.log_prob(action))

                # print(b_input.size())

                good_data_tensor = b_input.clone().detach()
                good_label_tensor = b_label.clone().detach()
                data_filter_tensor = action_bool.clone().detach()


                print(f'size of filterd data for {idx}th step of DVRL is : {good_data_tensor[data_filter_tensor].size()}')
                print(f'size of filterd label for {idx}th step of DVRL is : {good_label_tensor[data_filter_tensor].size()}')

                action_4_step = [torch.cat((RL_td_rest, good_data_tensor[data_filter_tensor]), dim=0)
                    , torch.cat((RL_tl_rest, good_label_tensor[data_filter_tensor]), dim=0)]

                reward, done, info = self.step(action=action_4_step)

                if self.useDiff == True:
                    reward = reward - REWARD_ORI

                self.R_lst.append(reward)
                self.total_reward_lst_trn.append(reward)

                Return = 0
                policy_loss = []
                Returns = []


                for r in self.R_lst[::-1]:
                    Return = r + self.gamma * Return
                    Returns.insert(0, Return)
                Returns = torch.tensor(Returns)
                print(f'Returns : {Returns}')
                # if self.reward_normalize == True:
                #     Returns = (Returns - Returns.mean()) / (Returns.std() + self.eps)

                for log_prob in self.policy_saved_log_probs_lst:
                    policy_loss.append(-log_prob * Returns)

                policy_loss = torch.sum(torch.cat(policy_loss))
                self.loss_lst_trn.append(float(policy_loss.item()))
                self.optimizerDVRL.zero_grad()
                policy_loss.backward()
                self.optimizerDVRL.step()
                print('gradient optimization done')

                print(f'self.loss_lst_trn is : {self.loss_lst_trn}')
                print(f'self.total_rwd_lst_trn is : {self.total_reward_lst_trn}')

                fig = plt.figure()
                ax1 = fig.add_subplot(12, 2, 1)
                ax1.plot(range(len(self.loss_lst_trn)), self.loss_lst_trn)
                ax1.set_title('DVRL loss')
                ax2 = fig.add_subplot(12, 2, 2)
                ax2.plot(range(len(self.total_reward_lst_trn)), self.total_reward_lst_trn)
                ax2.set_title('DVRL rwd')
                ax3 = fig.add_subplot(12, 2, 3)
                ax3.plot(range(len(self.lossLstLossGBASE)), self.lossLstLossGBASE)
                ax3.set_title('GBASE loss')
                ax4 = fig.add_subplot(12, 2, 4)
                ax4.plot(range(len(self.lossLstLossDBASE)), self.lossLstLossDBASE)
                ax4.set_title('DBASE loss')
                ax5 = fig.add_subplot(12, 2, 5)
                ax5.plot(range(len(self.lossLstLossG_GAN)), self.lossLstLossG_GAN)
                ax5.set_title('G_GAN loss')
                ax6 = fig.add_subplot(12, 2, 6)
                ax6.plot(range(len(self.lossLstLossG_DVRL)), self.lossLstLossG_DVRL)
                ax6.set_title('G_DVRL loss')
                ax7 = fig.add_subplot(12, 2, 7)
                ax7.plot(range(len(self.lossLstLossD)), self.lossLstLossD)
                ax7.set_title('D loss')
                ax8 = fig.add_subplot(12, 2, 8)
                ax8.plot(range(len(self.ProbFakeLstBASE)), self.ProbFakeLstBASE)
                ax8.set_title('Base Pfake')
                ax9 = fig.add_subplot(12, 2, 9)
                ax9.plot(range(len(self.ProbRealLstBASE)), self.ProbRealLstBASE)
                ax9.set_title('BASE Preal')
                ax10 = fig.add_subplot(12, 2, 10)
                ax10.plot(range(len(self.ProbFakeLst)), self.ProbFakeLst)
                ax10.set_title('Pfake')
                ax11 = fig.add_subplot(12, 2, 11)
                ax11.plot(range(len(self.ProbRealLst)), self.ProbRealLst)
                ax11.set_title('Preal')

                plt.savefig(self.test_fle_down_path + 'TRAIN_TOTAL_RESULT_PLOT.png', dpi=200)
                print('saving plot for DVRL complete!')
                plt.close()

                self.flush_lst()

        torch.set_grad_enabled(False)
        self.REINFORCE_GAN_G.eval()
        self.REINFORCE_GAN_D.eval()
        self.REINFORCE_GAN_GBASE.eval()
        self.REINFORCE_GAN_DBASE.eval()
        self.REINFORCE_DVRL.eval()

    def validation_step(self, vd2gen,vl2gen,RL_td_rest,RL_tl_rest):

        self.REINFORCE_GAN_G.eval()
        self.REINFORCE_GAN_D.eval()
        self.REINFORCE_GAN_GBASE.eval()
        self.REINFORCE_GAN_DBASE.eval()
        self.REINFORCE_DVRL.eval()

        for val_num2gen in self.val_num2genLst:

            print(f'getting baseline reward for {val_num2gen}th start getting baseline reward for {val_num2gen}th start ')
            print(f'getting baseline reward for {val_num2gen}th start getting baseline reward for {val_num2gen}th start')
            print(f'getting baseline reward for {val_num2gen}th start getting baseline reward for {val_num2gen}th start')

            with torch.set_grad_enabled(False):

                for i in range(val_num2gen):

                    noiseZ = self.get_gaussianNoise_z(self.rl_b_size)

                    GeneratedImg = self.REINFORCE_GAN_GBASE(noiseZ)

                    if i == 0:
                        RL_td_zero = vd2gen
                    else:
                        RL_td_zero = torch.cat((RL_td_zero,GeneratedImg),dim=0)


                RL_tl_zero = torch.zeros(RL_td_zero.size(0))

                with torch.set_grad_enabled(False):


                    RL_tl_rest = torch.ones_like(RL_tl_rest)

                    input_data4BASE = torch.cat((RL_td_rest, RL_td_zero), dim=0)
                    input_label4BASE = torch.cat((RL_tl_rest, RL_tl_zero), dim=0).long()

                    BASE_model_part = Prediction_lit_4REINFORCE1(save_dir=self.test_fle_down_path,
                                                                save_range=10,
                                                                beta4f1=self.beta4f1)

                    dm4BASE = datamodule_4REINFORCE1(batch_size=self.theta_b_size,
                                                    total_tdata=input_data4BASE,
                                                    total_tlabel=input_label4BASE,
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

                    trainer_part4BASE.fit(BASE_model_part, dm4BASE)
                    trainer_part4BASE.validate(BASE_model_part, dm4BASE)

                    REWARD_BASE = BASE_model_part.avg_acc_lst_val_f1score[-1]
                    self.reward4plotGBaseLst.append(REWARD_BASE)

            # del RL_tl_zero
            # del RL_td_zero
            # del RL_tl_rest
            # del RL_td_rest
            del input_label4BASE
            del input_data4BASE
            del BASE_model_part
            del dm4BASE
            del trainer_part4BASE

            print(f'getting baseline reward for {val_num2gen}th complete getting baseline reward for {val_num2gen}th complete ')
            print(
                f'getting baseline reward for {val_num2gen}th complete getting baseline reward for {val_num2gen}th complete ')
            print(
                f'getting baseline reward for {val_num2gen}th complete getting baseline reward for {val_num2gen}th complete ')

            print('')
            print('')
            print('')

            print(f'getting real reward for {val_num2gen} start getting real reward for {val_num2gen} start ')
            print(f'getting real reward for {val_num2gen} start getting real reward for {val_num2gen} start ')
            print(f'getting real reward for {val_num2gen} start getting real reward for {val_num2gen} start ')

            with torch.set_grad_enabled(False):

                for i in range(val_num2gen):

                    noiseZ = self.get_gaussianNoise_z(self.rl_b_size)

                    GeneratedImg = self.REINFORCE_GAN_G(noiseZ)

                    if i == 0:
                        RL_td_zero = vd2gen
                    else:
                        RL_td_zero = torch.cat((RL_td_zero, GeneratedImg), dim=0)

                RL_tl_zero = torch.zeros(RL_td_zero.size(0))

                with torch.set_grad_enabled(False):


                    RL_tl_rest = torch.ones_like(RL_tl_rest)

                    input_data4G = torch.cat((RL_td_rest, RL_td_zero), dim=0)
                    input_label4G = torch.cat((RL_tl_rest, RL_tl_zero), dim=0).long()

                    Gmodel_part = Prediction_lit_4REINFORCE1(save_dir=self.test_fle_down_path,
                                                                save_range=10,
                                                                beta4f1=self.beta4f1)

                    dm4G = datamodule_4REINFORCE1(batch_size=self.theta_b_size,
                                                    total_tdata=input_data4G,
                                                    total_tlabel=input_label4G,
                                                    val_data=self.val_data,
                                                    val_label=self.val_label)

                    trainer_part4G = pl.Trainer(max_steps=self.INNER_MAX_STEP,
                                                  max_epochs=1,
                                                  gpus=self.theta_gpu_num,
                                                  strategy='dp',
                                                  logger=False,
                                                  enable_checkpointing=False,
                                                  num_sanity_val_steps=0,
                                                  enable_model_summary=None)

                    trainer_part4G.fit(Gmodel_part, dm4G)
                    trainer_part4G.validate(Gmodel_part, dm4G)

                    REWARD_G = Gmodel_part.avg_acc_lst_val_f1score[-1]

                    self.reward4plotGLst.append(REWARD_G)

            # del RL_tl_zero
            # del RL_td_zero
            # del RL_tl_rest
            # del RL_td_rest
            del input_data4G
            del input_label4G
            del Gmodel_part
            del dm4G
            del trainer_part4G

            print(f'getting real reward for {val_num2gen}th complete getting real reward for {val_num2gen}th complete ')
            print(f'getting real reward for {val_num2gen}th complete getting real reward for {val_num2gen}th complete ')
            print(f'getting real reward for {val_num2gen}th complete getting real reward for {val_num2gen}th complete ')
            print('')
            print('')
            print('')

        torch.set_grad_enabled(True)
        self.REINFORCE_GAN_G.train()
        self.REINFORCE_GAN_D.train()
        self.REINFORCE_GAN_GBASE.train()
        self.REINFORCE_GAN_DBASE.train()
        self.REINFORCE_DVRL.train()

    def STARTTRNANDVAL(self,data,label,RL_td_rest,RL_tl_rest):
        print('===========================================================================================')
        print('===========================================================================================')
        print('===========================================================================================')
        print('===========================================================================================')
        print('===========================================================================================')
        print('start training step....')
        print('start training step....')
        print('start training step....')
        self.training_step(td2gen=data,tl2gen=label,RL_td_rest=RL_td_rest,RL_tl_rest=RL_tl_rest)
        print('training step complete!!!')
        print('validation step start...')
        print('validation step start...')
        print('validation step start...')
        print('')
        print('')
        print('')
        self.validation_step(vd2gen=data,vl2gen=label,RL_td_rest=RL_td_rest,RL_tl_rest=RL_tl_rest)
        print('validation step complete!!!')

        plt.plot(range(len(self.reward4plotGBaseLst)),self.reward4plotGBaseLst,'r')
        plt.plot(range(len(self.reward4plotGLst)),self.reward4plotGLst,'b')
        plt.xlabel('Generated Img Number')
        plt.ylabel('Validation F1 Beta Score')

        plt.savefig(self.test_fle_down_path + 'REWARD_COMPARE_RESULT.png', dpi=200)
        print('saving plot complete!')
        plt.close()
        print('===========================================================================================')
        print('===========================================================================================')
        print('===========================================================================================')
        print('===========================================================================================')
        print('===========================================================================================')


















