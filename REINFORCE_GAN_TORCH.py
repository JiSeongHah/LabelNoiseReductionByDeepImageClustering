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
# import gc
import tracemalloc

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
                 Num2Mul,
                 useDiff,
                 lr_G,
                 lr_D,
                 DVRL_INTERVAL,
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
        self.Num2Mul = Num2Mul
        self.DVRL_INTERVAL = DVRL_INTERVAL

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
        self.loss_lst_trn_tmp = []
        self.loss_lst_val = []

        self.policy_saved_log_probs_lst = []
        self.R_lst = []
        self.policy_saved_log_probs_lst_val = []
        self.R_lst_val = []
        self.theta_gpu_num = theta_gpu_num
        self.total_reward_lst_trn = []
        self.total_reward_lst_trn_tmp = []

        self.automatic_optimization = False
        self.gamma = gamma
        self.eps = eps
        self.rl_lr = rl_lr
        self.lr_G = lr_G
        self.lr_D = lr_D
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

        self.lossLstLossG_GAN_TMP = []
        self.lossLstLossG_DVRL_TMP = []
        self.lossLstLossD_TMP = []
        self.lossLstLossGBASE_TMP = []
        self.lossLstLossDBASE_TMP = []

        self.lossLstLossG_GAN = []
        self.lossLstLossG_DVRL = []
        self.lossLstLossD = []
        self.lossLstLossGBASE = []
        self.lossLstLossDBASE = []

        self.ProbFakeLst_TMP = []
        self.ProbRealLst_TMP = []
        self.ProbFakeLstBASE_TMP = []
        self.ProbRealLstBASE_TMP = []

        self.ProbFakeLst = []
        self.ProbRealLst = []
        self.ProbFakeLstBASE = []
        self.ProbRealLstBASE = []

        self.reward4plotGBaseLst =[]
        self.reward4plotGLst = []
        self.reward4plotGBaseLstAvg = []
        self.reward4plotGLstAvg = []

        self.reward4plotGBaseFiledLst = []
        self.reward4plotGFiledLst = []
        self.reward4plotGBaseFiledLstAvg = []
        self.reward4plotGFiledLstAvg = []



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
                              lr=self.lr_D,  # 학습률
                              eps=self.eps  # 0으로 나누는 것을 방지하기 위한 epsilon 값
                              )

        self.optimizerG = Adam(self.REINFORCE_GAN_G.parameters(),
                               lr=self.lr_G,  # 학습률
                               eps=self.eps  # 0으로 나누는 것을 방지하기 위한 epsilon 값
                               )
        self.optimizerGBASE = Adam(self.REINFORCE_GAN_GBASE.parameters(),
                               lr=self.lr_G,  # 학습률
                               eps=self.eps  # 0으로 나누는 것을 방지하기 위한 epsilon 값
                               )
        self.optimizerDBASE = Adam(self.REINFORCE_GAN_DBASE.parameters(),
                               lr=self.lr_D,  # 학습률
                               eps=self.eps  # 0으로 나누는 것을 방지하기 위한 epsilon 값
                               )
        self.optimizerDVRL = Adam(self.REINFORCE_DVRL.parameters(),
                               lr=self.rl_lr,  # 학습률
                               eps=self.eps  # 0으로 나누는 것을 방지하기 위한 epsilon 값
                               )

        ##########################VARS for RL model##################################

        self.theta_model_part = Prediction_lit_4REINFORCE1(save_dir=self.test_fle_down_path,
                                                      save_range=10,
                                                      beta4f1=self.beta4f1)
        


        test_dataset = MNIST(self.test_fle_down_path, train=False, download=True)

    def forwardGAN(self, x):

        GeneratedImages = self.REINFORCE_GAN_G(x)

        return GeneratedImages

    def forward(self,x):
        # print(f'RL part input is in device : {x.device}')
        probs_softmax = F.softmax(self.REINFORCE_DVRL(x.float()),dim=1)
        # print(f'example of prob_softmax is : {probs_softmax[:10]}')
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

    def flush_innerDVRL_lst(self):

        self.policy_saved_log_probs_lst = []
        self.policy_saved_log_probs_lst_val = []
        self.R_lst = []
        self.R_lst_val = []

        print('flushing lst on pl level complete')

    def flush_gan_lst(self):

        self.lossLstLossG_GAN_TMP.clear()
        self.lossLstLossG_DVRL_TMP.clear()
        self.lossLstLossD_TMP.clear()
        self.lossLstLossGBASE_TMP.clear()
        self.lossLstLossDBASE_TMP.clear()

        self.ProbFakeLst_TMP.clear()
        self.ProbRealLst_TMP.clear()
        self.ProbFakeLstBASE_TMP.clear()
        self.ProbRealLstBASE_TMP.clear()

    def flush_DVRL_lst(self):

        self.total_reward_lst_trn_tmp.clear()
        self.loss_lst_trn_tmp.clear()

    def flush_val_reward_lst(self):

        self.reward4plotGBaseLst.clear()
        self.reward4plotGLst.clear()
        self.reward4plotGBaseFiledLst.clear()
        self.reward4plotGFiledLst.clear()


    def imshow(self,img,saveDir):
        img = (img + 1) / 2
        img = img.squeeze()
        np_img = img.numpy()
        plt.imshow(np_img, cmap='gray')
        # plt.show()
        plt.savefig(saveDir + 'GENERATED_IMAGE.png', dpi=200)
        plt.cla()
        plt.clf()
        plt.close()

    def imshow_grid(self,img,saveDir,showNum,plotNum):
        img = utils.make_grid(img.cpu().detach())
        img = (img + 1) / 2
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

        randNum = random.random()
        # if randNum < 0.00001 and showNum % 10 ==0:
        #     plt.show()
        plt.savefig(saveDir + 'GENERATED_IMAGE_GRID'+str(plotNum)+'.png', dpi=200)
        plt.cla()
        plt.clf()
        plt.close()

        imshowDone = 'imshowDone'

        return imshowDone

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
                print(f'lossFake : {lossFake} with pFake : {torch.mean(p_fake)} '
                      f'and lossReal : {lossReal} with pReal : {torch.mean(p_real)}')

                lossD = (lossReal + lossFake).mean()

                return lossD
            if GorD == 'G':
                lossG = 0.5 * torch.mean(torch.square(p_fake-self.lsganC))

                print(f'lossG is : {lossG} with PFake : {torch.mean(p_fake)}')

                return lossG

    def step(self, action,val=False,PLOT=True):

        inputs_data4Theta, inputs_label4Theta = action[0].clone().detach(), action[1].clone().detach()
        inputs_label4Theta = inputs_label4Theta.clone().detach().long()

        

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

        if val == True:

            theta_model_part4val = Prediction_lit_4REINFORCE1(save_dir=self.test_fle_down_path,
                                                      save_range=10,
                                                      beta4f1=self.beta4f1)

            time2 = time.time()
            print('-----------------------------------------------------------------------------------')

            trainer_part.fit(theta_model_part4val, dm4Theta)

            trainer_part.validate(theta_model_part4val,dm4Theta)

            del trainer_part
            theta_model_part4val.flush_lst()

            if PLOT == True:
                pass


                # fig = plt.figure()
                # ax1 = fig.add_subplot(2, 4, 1)
                # ax1.plot(range(len(theta_model_part.avg_loss_lst_trn)), theta_model_part.avg_loss_lst_trn)
                # ax1.set_title('train loss')
                # ax2 = fig.add_subplot(2, 4, 2)
                # ax2.plot(range(len(theta_model_part.avg_acc_lst_trn_PRECISION)), theta_model_part.avg_acc_lst_trn_PRECISION)
                # ax2.set_title('train PRECISION')
                # ax3 = fig.add_subplot(2, 4, 3)
                # ax3.plot(range(len(theta_model_part.avg_acc_lst_trn_RECALL)), theta_model_part.avg_acc_lst_trn_RECALL)
                # ax3.set_title('train RECALL')
                # ax4 = fig.add_subplot(2, 4, 4)
                # ax4.plot(range(len(theta_model_part.avg_acc_lst_trn_f1score)), theta_model_part.avg_acc_lst_trn_f1score)
                # ax4.set_title('train F1 SCORE')
                #
                # ax5 = fig.add_subplot(2, 4, 5)
                # ax5.plot(range(len(theta_model_part.avg_loss_lst_val)), theta_model_part.avg_loss_lst_val)
                # ax5.set_title('val loss')
                # ax6 = fig.add_subplot(2, 4, 6)
                # ax6.plot(range(len(theta_model_part.avg_acc_lst_val_PRECISION)), theta_model_part.avg_acc_lst_val_PRECISION)
                # ax6.set_title('val PRECISION')
                # ax7 = fig.add_subplot(2, 4, 7)
                # ax7.plot(range(len(theta_model_part.avg_acc_lst_val_RECALL)), theta_model_part.avg_acc_lst_val_RECALL)
                # ax7.set_title('val RECALL')
                # ax8 = fig.add_subplot(2, 4, 8)
                # ax8.plot(range(len(theta_model_part.avg_acc_lst_val_f1score)), theta_model_part.avg_acc_lst_val_f1score)
                # ax8.set_title('val F1 SCORE')
                #
                # plt.savefig(self.test_fle_down_path + 'inner_model_result.png', dpi=200)
                # print('saving inner plot complete!')
                # plt.close()


            if self.reward_method == 'mean':
                reward = -1+ copy.deepcopy(float(np.mean(theta_model_part4val.avg_acc_lst_val_f1score[-self.conv_crit_num:])))
            if self.reward_method == 'last':
                reward = -1+ float(theta_model_part4val.avg_acc_lst_val_f1score[-1:][0])



            done = True
            info = 'step complete'

            dm4Theta.DelEveryVar()

            theta_model_part4val.DelEveryVar()

            del theta_model_part4val
            del dm4Theta
            del inputs_data4Theta
            del inputs_label4Theta
            print('----------------------------------------------------------------------------------------')
            print('')

            # gc.collect()

            return reward, done, info


        if val == False:


            time2 = time.time()
            print('-----------------------------------------------------------------------------------')

            trainer_part.fit(self.theta_model_part, dm4Theta)

            trainer_part.validate(self.theta_model_part,dm4Theta)

            del trainer_part
            self.theta_model_part.flush_lst()

            if PLOT == True:
                pass


                # fig = plt.figure()
                # ax1 = fig.add_subplot(2, 4, 1)
                # ax1.plot(range(len(theta_model_part.avg_loss_lst_trn)), theta_model_part.avg_loss_lst_trn)
                # ax1.set_title('train loss')
                # ax2 = fig.add_subplot(2, 4, 2)
                # ax2.plot(range(len(theta_model_part.avg_acc_lst_trn_PRECISION)), theta_model_part.avg_acc_lst_trn_PRECISION)
                # ax2.set_title('train PRECISION')
                # ax3 = fig.add_subplot(2, 4, 3)
                # ax3.plot(range(len(theta_model_part.avg_acc_lst_trn_RECALL)), theta_model_part.avg_acc_lst_trn_RECALL)
                # ax3.set_title('train RECALL')
                # ax4 = fig.add_subplot(2, 4, 4)
                # ax4.plot(range(len(theta_model_part.avg_acc_lst_trn_f1score)), theta_model_part.avg_acc_lst_trn_f1score)
                # ax4.set_title('train F1 SCORE')
                #
                # ax5 = fig.add_subplot(2, 4, 5)
                # ax5.plot(range(len(theta_model_part.avg_loss_lst_val)), theta_model_part.avg_loss_lst_val)
                # ax5.set_title('val loss')
                # ax6 = fig.add_subplot(2, 4, 6)
                # ax6.plot(range(len(theta_model_part.avg_acc_lst_val_PRECISION)), theta_model_part.avg_acc_lst_val_PRECISION)
                # ax6.set_title('val PRECISION')
                # ax7 = fig.add_subplot(2, 4, 7)
                # ax7.plot(range(len(theta_model_part.avg_acc_lst_val_RECALL)), theta_model_part.avg_acc_lst_val_RECALL)
                # ax7.set_title('val RECALL')
                # ax8 = fig.add_subplot(2, 4, 8)
                # ax8.plot(range(len(theta_model_part.avg_acc_lst_val_f1score)), theta_model_part.avg_acc_lst_val_f1score)
                # ax8.set_title('val F1 SCORE')
                #
                # plt.savefig(self.test_fle_down_path + 'inner_model_result.png', dpi=200)
                # print('saving inner plot complete!')
                # plt.close()


            if self.reward_method == 'mean':
                reward = -1+ copy.deepcopy(float(np.mean(self.theta_model_part.avg_acc_lst_val_f1score[-self.conv_crit_num:])))
            if self.reward_method == 'last':
                reward = -1+ float(self.theta_model_part.avg_acc_lst_val_f1score[-1:][0])



            done = True
            info = 'step complete'

            dm4Theta.DelEveryVar()

            del dm4Theta
            del inputs_data4Theta
            del inputs_label4Theta
            print('----------------------------------------------------------------------------------------')
            print('')

            # gc.collect()

            return reward, done, info


    def training_step(self, td2gen, tl2gen,RL_td_rest,RL_tl_rest,trainingNum):

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

                self.lossLstLossGBASE_TMP.append(lossG.item())
                self.lossLstLossDBASE_TMP.append(lossD.item())
                self.ProbFakeLstBASE_TMP.append(torch.mean(p_fake).item())
                self.ProbRealLstBASE_TMP.append(torch.mean(p_real).item())
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
        for param in self.REINFORCE_GAN_G.parameters():
            param.requires_grad = True

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
            selectionProb = self(GeneratedImgbyG)[:,1]

            lossG_GAN = self.GAN_LOSS(p_fake=p_fake,GorD='G')
            lossG_DVRL = torch.mean(torch.square(selectionProb-1.0))
            totalLossG = lossG_GAN + lossG_DVRL
            totalLossG.backward()

            self.lossLstLossG_GAN_TMP.append(lossG_GAN.item())
            self.lossLstLossG_DVRL_TMP.append(lossG_DVRL.item())
            self.lossLstLossD_TMP.append(lossD.item())
            self.ProbFakeLst_TMP.append(torch.mean(p_fake).item())
            self.ProbRealLst_TMP.append(torch.mean(p_real).item())
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
            if trainingNum % self.DVRL_INTERVAL == 0:
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

        if trainingNum % self.DVRL_INTERVAL == 0:

            with torch.set_grad_enabled(False):

                for param in self.REINFORCE_GAN_G.parameters():
                    param.requires_grad = False

                for i in range(self.Num2Gen):

                    noiseZ4Gen = self.get_gaussianNoise_z(self.Num2Mul*self.gan_trn_bSize)
                    GeneratedImg = self.REINFORCE_GAN_G(noiseZ4Gen)

                    GeneratedImg = GeneratedImg.clone().detach()

                    if i ==0:
                        RL_td_zero = td2gen
                    else:
                        RL_td_zero = torch.cat((RL_td_zero,GeneratedImg.clone().detach()),dim=0)

            for param in self.REINFORCE_GAN_G.parameters():
                param.requires_grad = True

            RL_td_zero = RL_td_zero.clone().detach()

            print('DVRL train_dataloading.......')
            DVRL_train_data = TensorDataset(RL_td_zero.clone().detach(), torch.zeros(RL_td_zero.clone().detach().size(0)))
            DVRL_train_sampler = RandomSampler(DVRL_train_data)
            DVRL_train_dataloader = DataLoader(DVRL_train_data,
                                               sampler=DVRL_train_sampler,
                                               batch_size=self.rl_b_size,
                                               num_workers=2)
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

                    action_4_step = [torch.cat((RL_td_rest.clone().detach(), good_data_tensor[data_filter_tensor].clone().detach()), dim=0)
                        , torch.cat((RL_tl_rest.clone().detach(), good_label_tensor[data_filter_tensor].clone().detach()), dim=0)]

                    reward, done, info = self.step(action=action_4_step)
                    # reward, done, info = 1, 'done', 'info'

                    if self.useDiff == True:
                        reward = reward - REWARD_ORI

                    self.R_lst.append(reward)
                    self.total_reward_lst_trn_tmp.append(reward)
                    # self.total_reward_lst_trn_tmp.append(1)

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
                    self.loss_lst_trn_tmp.append(float(policy_loss.item()))
                    # self.loss_lst_trn_tmp.append(1)
                    self.optimizerDVRL.zero_grad()
                    policy_loss.backward()
                    self.optimizerDVRL.step()
                    self.optimizerDVRL.zero_grad()
                    policy_loss = []
                    print('gradient optimization done')

                    print(f'self.loss_lst_trn is : {self.loss_lst_trn}')
                    print(f'self.total_rwd_lst_trn is : {self.total_reward_lst_trn}')

                    self.flush_innerDVRL_lst()
                    print(f'self.loss_lst_trn_tmp is : {self.loss_lst_trn_tmp}')
                    print(f'self.total_reward_lst_tmp is : {self.total_reward_lst_trn_tmp}')
                    print(f'slef.R_lst is : {self.R_lst}')
                    print(f'self.policy list : {self.policy_saved_log_probs_lst}')

                del action_4_step
            self.loss_lst_trn.append(round(np.mean(self.loss_lst_trn_tmp),3))
            self.total_reward_lst_trn.append(round(np.mean(self.total_reward_lst_trn_tmp),3))
            self.loss_lst_trn.append(round(np.mean(self.loss_lst_trn_tmp),3))
            self.total_reward_lst_trn.append(round(np.mean(self.total_reward_lst_trn_tmp),3))
            self.flush_DVRL_lst()
            del DVRL_train_data
            del DVRL_train_sampler
            del DVRL_train_dataloader

        self.lossLstLossG_GAN.append(round(np.mean(self.lossLstLossG_GAN_TMP),3))
        self.lossLstLossG_DVRL.append(round(np.mean(self.lossLstLossG_DVRL_TMP),3))
        self.lossLstLossD.append(round(np.mean(self.lossLstLossD_TMP),3))
        self.lossLstLossGBASE.append(round(np.mean(self.lossLstLossGBASE_TMP),3))
        self.lossLstLossDBASE.append(round(np.mean(self.lossLstLossDBASE_TMP),3))

        self.ProbFakeLst.append(round(np.mean(self.ProbFakeLst_TMP),3))
        self.ProbRealLst.append(round(np.mean(self.ProbRealLst_TMP),3))
        self.ProbFakeLstBASE.append(round(np.mean(self.ProbFakeLstBASE_TMP),3))
        self.ProbRealLstBASE.append(round(np.mean(self.ProbRealLstBASE_TMP),3))
        self.flush_gan_lst()


        torch.set_grad_enabled(False)
        self.REINFORCE_GAN_G.eval()
        self.REINFORCE_GAN_D.eval()
        self.REINFORCE_GAN_GBASE.eval()
        self.REINFORCE_GAN_DBASE.eval()
        self.REINFORCE_DVRL.eval()

        # gc.collect()

        trainingDone = 'trainingDone'

        return trainingDone

    def validation_step(self, vd2gen,vl2gen,RL_td_rest,RL_tl_rest,valNum):

        self.REINFORCE_GAN_G.eval()
        self.REINFORCE_GAN_D.eval()
        self.REINFORCE_GAN_GBASE.eval()
        self.REINFORCE_GAN_DBASE.eval()
        self.REINFORCE_DVRL.eval()

        for param in self.REINFORCE_DVRL.parameters():
            param.requires_grad = False

        for param in self.REINFORCE_GAN_GBASE.parameters():
            param.requires_grad = False

        for param in self.REINFORCE_GAN_G.parameters():
            param.requires_grad = False


        for val_num2gen in self.val_num2genLst:

            print(f'getting baseline reward for {val_num2gen}th start getting baseline reward for {val_num2gen}th start ')
            print(f'getting baseline reward for {val_num2gen}th start getting baseline reward for {val_num2gen}th start')
            print(f'getting baseline reward for {val_num2gen}th start getting baseline reward for {val_num2gen}th start')

            with torch.set_grad_enabled(False):

                noiseZ = self.get_gaussianNoise_z(self.Num2Mul*self.gan_trn_bSize*val_num2gen)

                GeneratedImg = self.REINFORCE_GAN_GBASE(noiseZ)

                if val_num2gen == 1:
                    imshowDone= self.imshow_grid(img=255.0*GeneratedImg[:5],saveDir=self.test_fle_down_path+'BASE_',showNum=1,plotNum=valNum)

                RL_td_zero = torch.cat((vd2gen,GeneratedImg.clone().detach()),dim=0)

                RL_td_zero = RL_td_zero.clone().detach()
                RL_tl_zero = torch.zeros(RL_td_zero.size(0))


                with torch.set_grad_enabled(False):


                    RL_tl_rest = torch.ones_like(RL_tl_rest)

                    input_data4BASE = torch.cat((RL_td_rest, RL_td_zero), dim=0)
                    input_label4BASE = torch.cat((RL_tl_rest, RL_tl_zero), dim=0).long()

                    action4step = [input_data4BASE,input_label4BASE]

                    REWARD_BASE,_,_ = self.step(action=action4step,PLOT= False,val=True)
                    self.reward4plotGBaseLst.append(round(float(REWARD_BASE),3))


                DVRL_val_data = TensorDataset(RL_td_zero.clone().detach(), torch.zeros(RL_td_zero.clone().detach().size(0)))
                DVRL_val_sampler = SequentialSampler(DVRL_val_data)
                DVRL_val_dataloader = DataLoader(DVRL_val_data,
                                                   sampler=DVRL_val_sampler,
                                                   batch_size=self.rl_b_size,
                                                   num_workers=2)
                print('DVRL train_dataloading complete!')
                print(f'size of input before filtered is : {RL_td_zero.size()} ')

                for idx, (b_input, b_label) in enumerate(DVRL_val_dataloader):
                    print(f'size of b_input for {idx} th step of DVRL is : {b_input.size()}')

                    action_probs = self.forward(b_input)
                    m = Categorical(action_probs)
                    action = m.sample()
                    action_bool = action.clone().detach().bool()

                    if idx ==0:
                        good_data_tensor = b_input.clone().detach()
                        good_label_tensor = b_label.clone().detach()
                        data_filter_tensor = action_bool.clone().detach()
                    else:
                        good_data_tensor = torch.cat((good_data_tensor,b_input.clone().detach()),dim=0)
                        good_label_tensor = torch.cat((good_label_tensor,b_label.clone().detach()),dim=0)
                        data_filter_tensor = torch.cat((data_filter_tensor,action_bool.clone().detach()),dim=0)

                print(f'size of input after filtered : {good_data_tensor[data_filter_tensor].size()}')

                action4step = [torch.cat((RL_td_rest.clone().detach(), good_data_tensor[data_filter_tensor].clone().detach()), dim=0),
                 torch.cat((RL_tl_rest.clone().detach(), good_label_tensor[data_filter_tensor].clone().detach()), dim=0).long()]

                REWARD_BASE_FILTERDVER,_,_ = self.step(action=action4step,PLOT=False,val=True)
                self.reward4plotGBaseFiledLst.append(round(float(REWARD_BASE_FILTERDVER),3))

            del RL_tl_zero
            del RL_td_zero
            del input_label4BASE
            del input_data4BASE



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

                noiseZ = noiseZ.clone().detach()

                GeneratedImg = self.REINFORCE_GAN_G(noiseZ)

                if val_num2gen == 1:
                    self.imshow_grid(img=255.0*GeneratedImg[:5],saveDir=self.test_fle_down_path+'COMPARE_',showNum=1,plotNum=valNum)


                RL_td_zero = torch.cat((vd2gen, GeneratedImg.clone().detach()), dim=0)

                RL_td_zero = RL_td_zero.clone().detach()
                RL_tl_zero = torch.zeros(RL_td_zero.size(0))

                with torch.set_grad_enabled(False):


                    RL_tl_rest = torch.ones_like(RL_tl_rest)

                    action4step = [torch.cat((RL_td_rest, RL_td_zero), dim=0),
                     torch.cat((RL_tl_rest, RL_tl_zero), dim=0).long()]

                    REWARD_G,_,_ = self.step(action=action4step,PLOT=False,val=True)

                    self.reward4plotGLst.append(round(float(REWARD_G),3))

                DVRL_val_data = TensorDataset(RL_td_zero.clone().detach(), torch.zeros(RL_td_zero.size(0)))
                DVRL_val_sampler = SequentialSampler(DVRL_val_data)
                DVRL_val_dataloader = DataLoader(DVRL_val_data,
                                                   sampler=DVRL_val_sampler,
                                                   batch_size=self.rl_b_size,
                                                   num_workers=2)
                print('DVRL validation _dataloading complete!')
                print(f'size of input before filtered is : {RL_td_zero.size()} ')

                for idx, (b_input, b_label) in enumerate(DVRL_val_dataloader):
                    print(f'size of b_input for {idx} th step of DVRL is : {b_input.size()}')

                    action_probs = self.forward(b_input)
                    m = Categorical(action_probs)
                    action = m.sample()
                    action_bool = action.clone().detach().bool()

                    if idx ==0:
                        good_data_tensor = b_input.clone().detach()
                        good_label_tensor = b_label.clone().detach()
                        data_filter_tensor = action_bool.clone().detach()
                    else:
                        good_data_tensor = torch.cat((good_data_tensor,b_input.clone().detach()),dim=0)
                        good_label_tensor = torch.cat((good_label_tensor,b_label.clone().detach()),dim=0)
                        data_filter_tensor = torch.cat((data_filter_tensor,action_bool.clone().detach()),dim=0)

                print(f'size of input after filtered : {good_data_tensor[data_filter_tensor].size()}')

                action4step = [torch.cat((RL_td_rest.clone().detach(), good_data_tensor[data_filter_tensor].clone().detach()), dim=0),
                 torch.cat((RL_tl_rest.clone().detach(), good_label_tensor[data_filter_tensor].clone().detach()), dim=0).long()]

                REWARD_G_FILTERDVER,_,_ = self.step(action=action4step,PLOT=False,val=True)
                self.reward4plotGFiledLst.append(round(float(REWARD_G_FILTERDVER),3))

            del RL_tl_zero
            del RL_td_zero
            del action4step


            print(f'getting real reward for {val_num2gen}th complete getting real reward for {val_num2gen}th complete ')
            print(f'getting real reward for {val_num2gen}th complete getting real reward for {val_num2gen}th complete ')
            print(f'getting real reward for {val_num2gen}th complete getting real reward for {val_num2gen}th complete ')
            print('')
            print('')
            print('')


        for param in self.REINFORCE_DVRL.parameters():
            param.requires_grad = True

        for param in self.REINFORCE_GAN_GBASE.parameters():
            param.requires_grad = True

        for param in self.REINFORCE_GAN_G.parameters():
            param.requires_grad = True

        del RL_tl_rest
        del RL_td_rest
        del DVRL_val_data
        del DVRL_val_sampler
        del DVRL_val_dataloader
        torch.set_grad_enabled(True)
        self.REINFORCE_GAN_G.train()
        self.REINFORCE_GAN_D.train()
        self.REINFORCE_GAN_GBASE.train()
        self.REINFORCE_GAN_DBASE.train()
        self.REINFORCE_DVRL.train()

        # gc.collect()

        validationDone = 'validationDone'

        return validationDone

    def STARTTRNANDVAL(self,data,label,RL_td_rest,RL_tl_rest,num2plot):
        # tracemalloc.start()
        #
        # snapshot1 = tracemalloc.take_snapshot()


        print('===========================================================================================')
        print('===========================================================================================')
        print('===========================================================================================')
        print('===========================================================================================')
        print('===========================================================================================')
        print('start training step....')
        print('start training step....')
        print('start training step....')
        trainingDone = self.training_step(td2gen=data,tl2gen=label,RL_td_rest=RL_td_rest,RL_tl_rest=RL_tl_rest,trainingNum=num2plot)
        print('training step complete!!!')

        if num2plot % self.DVRL_INTERVAL == 0:

            print('validation step start...')
            print('validation step start...')
            print('validation step start...')
            print('')
            print('')
            print('')

            validationDone = self.validation_step(vd2gen=data,vl2gen=label,RL_td_rest=RL_td_rest,RL_tl_rest=RL_tl_rest,valNum=num2plot)
            print('validation step complete!!!')

            print(f'len of what we goonna append is : {len(self.reward4plotGBaseLst[-len(self.val_num2genLst):])}')

            self.reward4plotGBaseLstAvg.append(np.mean(self.reward4plotGBaseLst[-len(self.val_num2genLst):]))
            self.reward4plotGLstAvg.append(np.mean(self.reward4plotGLst[-len(self.val_num2genLst):]))
            self.reward4plotGBaseFiledLstAvg.append(np.mean(self.reward4plotGBaseFiledLst[-len(self.val_num2genLst):]))
            self.reward4plotGFiledLstAvg.append(np.mean(self.reward4plotGFiledLst[-len(self.val_num2genLst):]))

            plt.plot(range(len(self.reward4plotGBaseLst)), self.reward4plotGBaseLst, 'r')
            plt.plot(range(len(self.reward4plotGLst)), self.reward4plotGLst, 'b')
            plt.xlabel('Generated Img Number')
            plt.ylabel('Validation F1 Beta Score')
            plt.savefig(self.test_fle_down_path + 'REWARD_COMPARE_RESULT_' + str(num2plot) + '.png', dpi=200)
            print('saving plot complete!')
            plt.cla()
            plt.clf()
            plt.close()

            self.reward4plotG_DiffLst = [G-GBase for G,GBase in zip(self.reward4plotGLst,self.reward4plotGBaseLst) ]
            plt.plot(range(len(self.reward4plotG_DiffLst)),self.reward4plotG_DiffLst)
            plt.xlabel('Generated Img Number')
            plt.ylabel('Validation F1 Beta Score')
            plt.savefig(self.test_fle_down_path + 'REWARD_COMPARE_RESULT_DIFF_' + str(num2plot) + '.png', dpi=200)
            print('saving plot complete!')
            plt.cla()
            plt.clf()
            plt.close()
            self.reward4plotG_DiffLst.clear()





            plt.plot(range(len(self.reward4plotGBaseFiledLst)), self.reward4plotGBaseFiledLst, 'r')
            plt.plot(range(len(self.reward4plotGFiledLst)), self.reward4plotGFiledLst, 'b')
            plt.xlabel('Generated Img Number')
            plt.ylabel('Validation F1 Beta Score')
            plt.savefig(self.test_fle_down_path + 'REWARD_COMPARE_RESULT_FILTEREDVER_' + str(num2plot) + '.png',
                        dpi=200)
            print('saving plot complete!')
            plt.cla()
            plt.clf()
            plt.close()

            self.reward4plotG_Filed_DiffLst = [G - GBase for G, GBase in zip(self.reward4plotGFiledLst, self.reward4plotGBaseFiledLst)]
            plt.plot(range(len(self.reward4plotG_Filed_DiffLst)), self.reward4plotG_Filed_DiffLst)
            plt.xlabel('Generated Img Number')
            plt.ylabel('Validation F1 Beta Score')
            plt.savefig(self.test_fle_down_path + 'REWARD_COMPARE_RESULT_DIFF_FILTEREDVER' + str(num2plot) + '.png', dpi=200)
            print('saving plot complete!')
            plt.cla()
            plt.clf()
            plt.close()
            self.reward4plotG_Filed_DiffLst.clear()









            plt.plot(range(len(self.reward4plotGBaseLstAvg)), self.reward4plotGBaseLstAvg, 'r')
            plt.plot(range(len(self.reward4plotGLstAvg)), self.reward4plotGLstAvg, 'b')
            plt.xlabel('Generated Img Number')
            plt.ylabel('Validation F1 Beta Score')
            plt.savefig(self.test_fle_down_path + 'MEAN_REWARD_COMPARE_RESULT_' + str(num2plot) + '.png', dpi=200)
            print('saving plot complete!')
            plt.cla()
            plt.clf()
            plt.close()

            self.reward4plotG_DiffLstAvg = [G - GBase for G, GBase in
                                               zip(self.reward4plotGLstAvg, self.reward4plotGBaseLstAvg)]
            plt.plot(range(len(self.reward4plotG_DiffLstAvg)), self.reward4plotG_DiffLstAvg)
            plt.xlabel('Generated Img Number')
            plt.ylabel('Validation F1 Beta Score')
            plt.savefig(self.test_fle_down_path + 'REWARD_COMPARE_RESULT_AVG_DIFF_' + str(num2plot) + '.png', dpi=200)
            print('saving plot complete!')
            plt.cla()
            plt.clf()
            plt.close()
            self.reward4plotG_DiffLstAvg.clear()









            plt.plot(range(len(self.reward4plotGBaseFiledLstAvg)), self.reward4plotGBaseFiledLstAvg, 'r')
            plt.plot(range(len(self.reward4plotGFiledLstAvg)), self.reward4plotGFiledLstAvg, 'b')
            plt.xlabel('Generated Img Number')
            plt.ylabel('Validation F1 Beta Score')
            plt.savefig(self.test_fle_down_path + 'MEAN_REWARD_COMPARE_RESULT_FILTEREDVER_' + str(num2plot) + '.png',
                        dpi=200)
            print('saving plot complete!')
            plt.cla()
            plt.clf()
            plt.close()


            self.reward4plotG_Filed_DiffLstAvg = [G - GBase for G, GBase in
                                               zip(self.reward4plotGFiledLstAvg, self.reward4plotGBaseFiledLstAvg)]
            plt.plot(range(len(self.reward4plotG_Filed_DiffLstAvg)), self.reward4plotG_Filed_DiffLstAvg)
            plt.xlabel('Generated Img Number')
            plt.ylabel('Validation F1 Beta Score')
            plt.savefig(self.test_fle_down_path + 'REWARD_COMPARE_RESULT_AVG_DIFF_FILTEREDVER' + str(num2plot) + '.png', dpi=200)
            print('saving plot complete!')
            plt.cla()
            plt.clf()
            plt.close()
            self.reward4plotG_Filed_DiffLstAvg.clear()





            self.flush_val_reward_lst()

        # with torch.set_grad_enabled(False):
        #
        #
        #
        #     noiseZ = self.get_gaussianNoise_z(self.Num2Mul*self.gan_trn_bSize)
        #
        #     GeneratedImg = self.REINFORCE_GAN_GBASE(noiseZ)
        #
        #     imshowDone= self.imshow_grid(img=255.0*GeneratedImg[:5],saveDir=self.test_fle_down_path+'BASE_',showNum=1,plotNum=num2plot)
        #
        #     noiseZ = self.get_gaussianNoise_z(self.Num2Mul * self.gan_trn_bSize)
        #
        #     GeneratedImg = self.REINFORCE_GAN_G(noiseZ)
        #
        #     imshowDone = self.imshow_grid(img=255.0*GeneratedImg[:5], saveDir=self.test_fle_down_path + 'COMPARE_', showNum=1,
        #                                   plotNum=num2plot)



        fig = plt.figure(constrained_layout=True)
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(range(len(self.loss_lst_trn)), self.loss_lst_trn)
        ax1.set_title('DVRL loss')
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(range(len(self.total_reward_lst_trn)), self.total_reward_lst_trn)
        ax2.set_title('DVRL rwd')
        plt.savefig(self.test_fle_down_path + 'DVRL_RESULT_PLOT.png', dpi=300)
        fig.clf()
        plt.cla()
        plt.clf()
        plt.close()

        fig = plt.figure(constrained_layout=True)
        ax3 = fig.add_subplot(1, 2, 1)
        ax3.plot(range(len(self.lossLstLossGBASE)), self.lossLstLossGBASE)
        ax3.set_title('GBASE loss')
        ax4 = fig.add_subplot(1, 2, 2)
        ax4.plot(range(len(self.lossLstLossDBASE)), self.lossLstLossDBASE)
        ax4.set_title('DBASE loss')
        plt.savefig(self.test_fle_down_path + 'BASE_RESULT_PLOT.png', dpi=300)
        fig.clf()
        plt.cla()
        plt.clf()
        plt.close()

        fig = plt.figure(constrained_layout=True)
        ax5 = fig.add_subplot(1, 3, 1)
        ax5.plot(range(len(self.lossLstLossG_GAN)), self.lossLstLossG_GAN)
        ax5.set_title('G_GAN loss')
        ax6 = fig.add_subplot(1, 3, 2)
        ax6.plot(range(len(self.lossLstLossG_DVRL)), self.lossLstLossG_DVRL)
        ax6.set_title('G_DVRL loss')
        ax7 = fig.add_subplot(1, 3, 3)
        ax7.plot(range(len(self.lossLstLossD)), self.lossLstLossD)
        ax7.set_title('D loss')
        plt.savefig(self.test_fle_down_path + 'GAN_LOSS_RESULT_PLOT.png', dpi=300)
        fig.clf()
        plt.cla()
        plt.clf()
        plt.close()

        fig = plt.figure(constrained_layout=True)
        ax8 = fig.add_subplot(1, 4, 1)
        ax8.plot(range(len(self.ProbFakeLstBASE)), self.ProbFakeLstBASE)
        ax8.set_title('Base Pfake')
        ax9 = fig.add_subplot(1, 4, 2)
        ax9.plot(range(len(self.ProbRealLstBASE)), self.ProbRealLstBASE)
        ax9.set_title('BASE Preal')
        ax10 = fig.add_subplot(1, 4, 3)
        ax10.plot(range(len(self.ProbFakeLst)), self.ProbFakeLst)
        ax10.set_title('Pfake')
        ax11 = fig.add_subplot(1, 4, 4)
        ax11.plot(range(len(self.ProbRealLst)), self.ProbRealLst)
        ax11.set_title('Preal')
        plt.savefig(self.test_fle_down_path + 'GAN_PROB_RESULT_PLOT.png', dpi=300)
        print('saving plot for DVRL complete!')
        fig.clf()
        plt.cla()
        plt.clf()
        plt.close()

        # snapshot2 = tracemalloc.take_snapshot()
        #
        # top_stats = snapshot2.compare_to(snapshot1, 'lineno')



        # print(f'printing top 10 memory leak')
        # print(f'printing top 10 memory leak')
        # print(f'printing top 10 memory leak')
        # print(f'printing top 10 memory leak')
        # print(f'printing top 10 memory leak')
        # print(f'printing top 10 memory leak')
        # print(f'printing top 10 memory leak')
        # for stat in top_stats[:10]:
        #     print(stat)
        #
        # top_stats = snapshot2.statistics('traceback')
        #
        # stat = top_stats[0]
        #
        # for line in stat.traceback.format():
        #     print(line)
        #
        #
        # tracemalloc.stop()



        print('===========================================================================================')
        print('===========================================================================================')
        print('===========================================================================================')
        print('===========================================================================================')
        print('===========================================================================================')


















