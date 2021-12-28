from save_funcs import mk_name,lst2csv,createDirectory
from EXCUTE_RL import EXCUTE_RL



if __name__ == '__main__':
    gamma = 0.999
    eps = 1e-9
    rl_lr = 4e-06
    rl_b_size = 1
    theta_b_size = 8192
    reward_normalize = True
    theta_stop_threshold = 0.01
    rl_stop_threshold = 0.01
    theta_gpu_num = [2]
    rwd_spread = True
    theta_max_epch = 25
    max_ep = 5000
    RL_save_range = 50
    conv_crit_num = 5
    inner_max_step = 11
    wayofdata = 'sum'
    beta4f1 = 100
    noise_ratio = 0.5
    split_ratio = int(5923*0.05)

    master_dir = '/home/a286/'

    specific_dir_name = mk_name(rwd_spread=rwd_spread,reward_normalize=reward_normalize,noise_ratio=noise_ratio,split_ratio=split_ratio,beta=1)

    test_fle_down_path = master_dir+'hjs_dir1/'+specific_dir_name +'/'
    trn_fle_down_path =  master_dir+'hjs_dir1/'+specific_dir_name + '/'
    model_save_load_path = master_dir+'hjs_dir1/'+specific_dir_name + '/'
    createDirectory(master_dir+'/hjs_dir1/'+specific_dir_name)

    do_it = EXCUTE_RL(gamma=gamma,eps=eps,rl_lr=rl_lr,rl_b_size=rl_b_size,theta_b_size=theta_b_size,reward_normalize=reward_normalize,
                 theta_stop_threshold=theta_stop_threshold,rl_stop_threshold=rl_stop_threshold,test_fle_down_path=test_fle_down_path,
                      trn_fle_down_path=trn_fle_down_path,theta_gpu_num=theta_gpu_num,model_save_load_path=model_save_load_path,rwd_spread=rwd_spread,
                      theta_max_epch=theta_max_epch,max_ep=max_ep,wayofdata=wayofdata,noise_ratio=noise_ratio,split_ratio=split_ratio,
                      beta4f1=beta4f1,inner_max_step=inner_max_step,conv_crit_num=conv_crit_num,RL_save_range=RL_save_range)

    excute_rl = do_it.excute_RL()





