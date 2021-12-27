from save_funcs import mk_name,lst2csv,createDirectory
from EXCUTE_RL import EXCUTE_RL


if __name__ == '__main__':
    gamma = 0.999
    eps = 1e-9
    rl_lr = 4e-06
    rl_b_size = 1
    theta_b_size = 1024
    reward_normalize = True
    theta_stop_threshold = 0.01
    rl_stop_threshold = 0.01
    theta_gpu_num = [3]
    rwd_spread = True
    theta_max_epch = 200
    max_ep = 50
    inner_max_step = 1
    wayofdata = 'sum'
    beta4f1 = 100
    noise_ratio = 1.3
    split_ratio = int(5923*0.05)

    specific_dir_name = mk_name(rwd_spread=rwd_spread,reward_normalize=reward_normalize,noise_ratio=noise_ratio,split_ratio=split_ratio,beta=1)

    test_fle_down_path = '/home/a286winteriscoming/hjs_dir1/'+specific_dir_name +'/'
    trn_fle_down_path = '/home/a286winteriscoming/hjs_dir1/'+specific_dir_name + '/'
    model_save_load_path = '/home/a286winteriscoming/hjs_dir1/'+specific_dir_name + '/'
    createDirectory('/home/a286winteriscoming/hjs_dir1/'+specific_dir_name)

    do_it = EXCUTE_RL(gamma=gamma,eps=eps,rl_lr=rl_lr,rl_b_size=rl_b_size,theta_b_size=theta_b_size,reward_normalize=reward_normalize,
                 theta_stop_threshold=theta_stop_threshold,rl_stop_threshold=rl_stop_threshold,test_fle_down_path=test_fle_down_path,
                      trn_fle_down_path=trn_fle_down_path,theta_gpu_num=theta_gpu_num,model_save_load_path=model_save_load_path,rwd_spread=rwd_spread,
                      theta_max_epch=theta_max_epch,max_ep=max_ep,wayofdata=wayofdata,noise_ratio=noise_ratio,split_ratio=split_ratio,
                      beta4f1=beta4f1,inner_max_step=inner_max_step)

    excute_rl = do_it.excute_RL()






