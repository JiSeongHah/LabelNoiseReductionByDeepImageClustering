import torch
from torchvision.datasets import MNIST
from MK_NOISED_DATA import mk_noisy_data
from EXCUTE_RL_GAN import EXCUTE_RL_GAN
from REINFORCE_GAN_TORCH import REINFORCE_GAN_TORCH
from save_funcs import load_my_model
import numpy as np
from save_funcs import mk_name,createDirectory


gamma = 0.999
eps = 1e-9
rl_lr = 4e-05
lr_G = 2e-4
lr_D = 3e-5
theta_b_size = 1024
reward_normalize = True
theta_stop_threshold = 0.01
rl_stop_threshold = 0.01
theta_gpu_num = [0]
rwd_spread = True
theta_max_epch = 25
max_ep = 5000
RL_save_range = 10
wayofdata = 'pureonly'
noise_ratio = 1
split_ratio = int(5923 * 0.05)
master_dir = '/home/a286winteriscoming/'
rl_b_size = 2000
reward_method = 'last'
dNoise = 100
dHidden = 256
gan_trn_bSize = 64
gan_val_bSize = 64
beta4f1 = 10
max_step_trn = 100 #deprecated
max_step_val = 100 #deprecated
whichGanLoss= 'lsgan'
INNER_MAX_STEP = 32
model_num_now = 0
GLoadNum = 0
GbaseLoadNum = 0
DLoadNum = 0
DVRLLoadNum = 0
val_num2genLst = [i+1 for i in range(5)]
Num2Gen = 32
useDiff = False
Num2Mul = 10
DVRL_INTERVAL_LST = [16]

lsganA = 0.1
lsganB = 0.9
lsganC = 1

for DVRL_INTERVAL in DVRL_INTERVAL_LST:

    specific_dir_name = mk_name(dirRL7='/',
                                whichGanLoss=whichGanLoss,
                                split_ratio=split_ratio,
                                beta=beta4f1,
                                dNoise=dNoise,
                                dHidden=dHidden,
                                Num2Gen=Num2Gen,
                                rlb=rl_b_size,
                                ganb=gan_trn_bSize,
                                INNMSTEP=INNER_MAX_STEP,
                                Num2Mul=Num2Mul,
                                DVRL_INTERVAL= DVRL_INTERVAL)

    test_fle_down_path = master_dir + 'hjs_dir1/' + specific_dir_name + '/'
    trn_fle_down_path = master_dir + 'hjs_dir1/' + specific_dir_name + '/'
    model_save_load_path = master_dir + 'hjs_dir1/' + specific_dir_name + '/Models/'
    createDirectory(master_dir + '/hjs_dir1/' + specific_dir_name)

    doIt = EXCUTE_RL_GAN(gamma = gamma,
                        eps = eps,
                        rl_lr = rl_lr,
                        model_num_now = model_num_now,
                        theta_b_size = theta_b_size,
                        reward_normalize = reward_normalize,
                        theta_stop_threshold = theta_stop_threshold,
                        rl_stop_threshold = rl_stop_threshold,
                        trn_fle_down_path=trn_fle_down_path,
                        test_fle_down_path=test_fle_down_path,
                        model_save_load_path=model_save_load_path,
                        reward_method =  reward_method,
                        theta_gpu_num = theta_gpu_num,
                        rwd_spread = rwd_spread,
                        theta_max_epch = theta_max_epch,
                        max_ep = max_ep,
                        RL_save_range = RL_save_range,
                        wayofdata = 'pureonly',
                        noise_ratio = noise_ratio,
                        split_ratio = split_ratio,
                        rl_b_size =rl_b_size,
                        dNoise = dNoise,
                        dHidden = dHidden,
                        INNER_MAX_STEP = INNER_MAX_STEP,
                        gan_trn_bSize = gan_trn_bSize,
                        gan_val_bSize = gan_val_bSize,
                        beta4f1 = beta4f1,
                        max_step_trn = max_step_trn, #deprecated,
                        max_step_val = max_step_val,#deprecated
                        whichGanLoss=whichGanLoss,
                        GLoadNum=GLoadNum,
                        GbaseLoadNum= GbaseLoadNum,
                        DLoadNum=DLoadNum,
                        DVRLLoadNum=DVRLLoadNum,
                        val_num2genLst= val_num2genLst,
                        Num2Gen = Num2Gen,
                        Num2Mul=Num2Mul,
                        useDiff= useDiff,
                        lr_G=lr_G,
                        lr_D=lr_D,
                        lsganA = lsganA,
                        lsganB = lsganB,
                        lsganC = lsganC,
                        DVRL_INTERVAL =DVRL_INTERVAL
                         )
    doIt.excute_RL(GLoadNum=GLoadNum,
                   DLoadNum=DLoadNum,
                   GbaseLoadNum=GbaseLoadNum,
                   DvrlLoadNum=DVRLLoadNum)