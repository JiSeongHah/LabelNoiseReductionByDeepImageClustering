from save_funcs import mk_name,lst2csv,createDirectory
from EXCUTE_RL import EXCUTE_RL
import torch
from torchvision.datasets import MNIST
from MK_NOISED_DATA import mk_noisy_data
from REINFORCE_TORCH import REINFORCE_TORCH
from save_funcs import load_my_model

if __name__ == '__main__':
    gamma = 0.999
    eps = 1e-9
    rl_lr = 4e-06
    theta_b_size = 1024
    reward_normalize = True
    theta_stop_threshold = 0.01
    rl_stop_threshold = 0.01
    theta_gpu_num = [3]
    rwd_spread = True
    theta_max_epch = 25
    max_ep = 5000
    RL_save_range = 100
    conv_crit_num = 5
    inner_max_step = 11
    wayofdata = 'sum'
    noise_ratio = 10
    split_ratio = int(5923*0.2)
    master_dir = '/home/a286/'
    rl_b_size = 128
    WINDOW = 20

    rl_b_size_lst = [16, 32, 64]
    beta4f1Lst = [100]
    INNER_MAX_STEP_Lst = [32]
    reward_method = 'last'

    for beta4f1 in beta4f1Lst:
        for INNER_MAX_STEP in INNER_MAX_STEP_Lst:
            for rl_b_size in rl_b_size_lst:
                specific_dir_name = mk_name(DVRL='/',dir6='/',innerNum=INNER_MAX_STEP,noise_ratio=noise_ratio,split_ratio=split_ratio,beta=beta4f1,rlSize=rl_b_size)

                test_fle_down_path = master_dir+'hjs_dir1/'+specific_dir_name +'/'
                trn_fle_down_path =  master_dir+'hjs_dir1/'+specific_dir_name + '/'
                model_save_load_path = master_dir+'hjs_dir1/'+specific_dir_name + '/'
                createDirectory(master_dir+'/hjs_dir1/'+specific_dir_name)

                do_it = EXCUTE_RL(gamma=gamma,
                                  eps=eps,
                                  rl_lr=rl_lr,
                                  rl_b_size=rl_b_size,
                                  theta_b_size=theta_b_size,
                                  reward_normalize=reward_normalize,
                                  theta_stop_threshold=theta_stop_threshold,
                                  rl_stop_threshold=rl_stop_threshold,
                                  test_fle_down_path=test_fle_down_path,
                                  trn_fle_down_path=trn_fle_down_path,
                                  theta_gpu_num=theta_gpu_num,
                                  model_save_load_path=model_save_load_path,
                                  rwd_spread=rwd_spread,
                                  theta_max_epch=theta_max_epch,
                                  max_ep=max_ep,
                                  wayofdata=wayofdata,
                                  noise_ratio=noise_ratio,
                                  split_ratio=split_ratio,
                                  beta4f1=beta4f1,
                                  INNER_MAX_STEP=INNER_MAX_STEP,
                                  reward_method=reward_method,
                                  inner_max_step=inner_max_step,
                                  conv_crit_num=conv_crit_num,
                                  WINDOW=WINDOW,
                                  RL_save_range=RL_save_range)

                excute_rl = do_it.excute_RL()




