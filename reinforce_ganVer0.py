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
rl_lr = 4e-06

theta_b_size = 8192
reward_normalize = True
theta_stop_threshold = 0.01
rl_stop_threshold = 0.01
theta_gpu_num = [0]
rwd_spread = True
theta_max_epch = 25
max_ep = 5000
RL_save_range = 100
wayofdata = 'pureonly'
noise_ratio = 1
split_ratio = int(5923 * 0.05)
master_dir = '/home/a286winteriscoming/'
rl_b_size = split_ratio
reward_method = 'nothing'
dNoise = 100
dHidden = 128
gan_trn_bSize = 32
gan_val_bSize = 32
beta4f1 = 1
max_step_trn = 100 #deprecated
max_step_val = 100 #deprecated


specific_dir_name = mk_name(dir2='/',reward_method=str(reward_method),noise_ratio=noise_ratio, split_ratio=split_ratio,
                            beta=beta4f1)

test_fle_down_path = master_dir + 'hjs_dir1/' + specific_dir_name + '/'
trn_fle_down_path = master_dir + 'hjs_dir1/' + specific_dir_name + '/'
model_save_load_path = master_dir + 'hjs_dir1/' + specific_dir_name + '/'
createDirectory(master_dir + '/hjs_dir1/' + specific_dir_name)

doIt = EXCUTE_RL_GAN(gamma = gamma,
                    eps = eps,
                    rl_lr = rl_lr,
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
                    gan_trn_bSize = gan_trn_bSize,
                    gan_val_bSize = gan_val_bSize,
                    beta4f1 = beta4f1,
                    max_step_trn = max_step_trn, #deprecated,
                    max_step_val = max_step_val) #deprecated)
doIt.excute_RL()