import torch
from torchvision.datasets import MNIST
from MK_NOISED_DATA import mk_noisy_data
from simple_torch import simple_torch
from save_funcs import load_my_model

class excute_simple_rl:
    def __init__(self,gamma,eps,rl_lr,rl_b_size,theta_b_size,reward_normalize,rwd_spread,inner_max_step,
                 theta_stop_threshold,rl_stop_threshold,test_fle_down_path,trn_fle_down_path,beta4f1,
                 theta_gpu_num,model_save_load_path,theta_max_epch,max_ep,wayofdata,noise_ratio,split_ratio,
                 conv_crit_num,RL_save_range,data_cut_num):

        ####################################VARS FOR CLASS : REINFORCE_TORCH ############################
        self.rl_b_size = rl_b_size
        self.theta_b_size = theta_b_size
        self.gamma = gamma
        self.rl_lr = rl_lr
        self.reward_normalize = reward_normalize

        self.test_fle_down_path = test_fle_down_path
        self.model_save_load_path = model_save_load_path
        self.theta_gpu_num = theta_gpu_num

        self.MAX_EP = max_ep
        self.theta_max_epch = theta_max_epch
        self.theta_stop_threshold = theta_stop_threshold
        self.rl_stop_threshold = rl_stop_threshold
        self.rwd_spread = rwd_spread
        self.beta4f1 = beta4f1
        self.inner_max_step = inner_max_step
        self.conv_crit_num = conv_crit_num
        self.RL_save_range = RL_save_range

        self.eps = eps
        self.data_cut_num = data_cut_num

        ####################################VARS FOR CLASS : REINFORCE_TORCH ############################
        self.noise_ratio = noise_ratio
        self.split_ratio = split_ratio
        self.wayofdata = wayofdata
        self.trn_fle_down_path = trn_fle_down_path
        ####################################VARS FOR CLASS : EXCUTE_RL ############################


    def excute_RL(self):

        RL_train_dataset = MNIST(self.trn_fle_down_path, train=True, download=True)
        RL_val_dataset = MNIST(self.trn_fle_down_path, train=False, download=True)

        RL_train_data = RL_train_dataset.data.numpy()
        RL_train_label = RL_train_dataset.targets.numpy()

        RL_train_label_zero = RL_train_label[RL_train_label == 0]
        RL_train_label_rest = RL_train_label[RL_train_label != 0]

        RL_train_data_zero = RL_train_data[RL_train_label == 0]
        RL_train_data_rest = RL_train_data[RL_train_label != 0]

        RL_val_inputs = torch.from_numpy(RL_val_dataset.data.numpy()).clone().detach().unsqueeze(1)
        RL_val_labels = torch.from_numpy(RL_val_dataset.targets.numpy()).clone().detach()


        if self.wayofdata == 'sum':
            RL_train_data_zero_little = torch.from_numpy(mk_noisy_data(raw_data=RL_train_data_zero, noise_ratio=self.noise_ratio,
                                                  split_ratio=self.split_ratio, way=self.wayofdata)).unsqueeze(1)
            RL_train_label_zero_little = torch.from_numpy(RL_train_label_zero[:])
        elif self.wayofdata == 'pureonly':
            RL_train_data_zero_little = torch.from_numpy(mk_noisy_data(raw_data=RL_train_data_zero, noise_ratio=self.noise_ratio,
                                                      split_ratio=self.split_ratio, way=self.wayofdata)).unsqueeze(1)
            RL_train_label_zero_little = torch.from_numpy(RL_train_label_zero[:self.split_ratio])
        elif self.wayofdata == 'noiseonly':
            RL_train_data_zero_little = torch.from_numpy(
                mk_noisy_data(raw_data=RL_train_data_zero, noise_ratio=self.noise_ratio,
                              split_ratio=self.split_ratio, way=self.wayofdata)).unsqueeze(1)
            RL_train_label_zero_little = torch.from_numpy(RL_train_label_zero[:])



        print('spliting train data done')
        print('start val data job....')

        print(f'shape of val_inputs is : {RL_val_inputs.shape}')
        print('spliting validation ddddata done')

        print('valid_dataloading.......')
        # validation_data = TensorDataset(RL_val_inputs, RL_val_labels)
        # validation_sampler = SequentialSampler(validation_data)
        # validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=self.RL_b_size,
        #                                    num_workers=4)

        del RL_train_dataset
        del RL_train_data
        del RL_val_dataset


        print('valid_dataloading done....')

        try:
            print('self.model_save_load_path is ',self.model_save_load_path)
            self.model_num_now = float((load_my_model(self.model_save_load_path).split('/')[-1].split('.')[0]))
            print('self.model_num_now is : ',self.model_num_now)
            print('testttttttttttttt : ',load_my_model(self.model_save_load_path))

            REINFORCE_START = torch.load(load_my_model(self.model_save_load_path))
            REINFORCE_START.model_num_now = self.model_num_now
            print(f'REINFORCE_START.model_num_now is : {REINFORCE_START.model_num_now}')
            print(f'len of REINFORCE_START.total_reward_lst_trn : {len(REINFORCE_START.total_reward_lst_trn)}')
            print('model loading doneeeeeeeeeeeeee')
            #time.sleep(5)
            print('successsuccesssuccesssuccesssuccesssuccesssuccesssuccess')
        except:
            print('model loading failed so loaded fresh model')
            REINFORCE_START = simple_torch(gamma=self.gamma, eps=self.eps, rl_lr=self.rl_lr,
                                              rl_b_size=self.rl_b_size, theta_b_size=self.theta_b_size,
                                              reward_normalize=self.reward_normalize, val_data=RL_val_inputs,
                                              val_label=RL_val_labels,
                                              theta_stop_threshold=self.theta_stop_threshold,
                                              rl_stop_threshold=self.rl_stop_threshold,
                                              test_fle_down_path=self.test_fle_down_path,
                                              theta_gpu_num=self.theta_gpu_num, rwd_spread=self.rwd_spread,
                                              model_save_load_path=self.model_save_load_path,
                                              theta_max_epch=self.theta_max_epch, max_ep=self.MAX_EP,
                                              beta4f1=self.beta4f1, inner_max_step=self.inner_max_step,
                                              conv_crit_num=self.conv_crit_num,
                                              data_cut_num=self.data_cut_num)
            REINFORCE_START.model_num_now = 0
            print('failedfailedfailedfailedfailedfailedfailedfailedfailedfailed')


        for i in range(10000):
            print(f'{i} th training RL start')

            REINFORCE_START.training_step(RL_td_zero=RL_train_data_zero_little, RL_tl_zero=RL_train_label_zero_little,
                                          RL_td_rest=RL_train_data_rest, RL_tl_rest=RL_train_label_rest, training_num=i)
            if i%self.RL_save_range ==0 and i!=0:
                try:
                    print(f'REINFORCE_START.model_num_now is : {REINFORCE_START.model_num_now}')
                    torch.save(REINFORCE_START,self.model_save_load_path+str(i+REINFORCE_START.model_num_now)+'.pt')
                    print('saving RL model complete')
                    print('saving RL model complete')
                    print('saving RL model complete')
                    print('saving RL model complete')
                    print('saving RL model complete')
                    print('saving RL model complete')
                except:
                    print('saving model failed')
            print(f'{i} th training for RL done')
            print('                   ')
            print('                   ')
            print('                   ')
            print('                   ')
            print('                   ')
            print('                   ')
            print('                   ')
            print('                   ')
            #print(f' reward lst is : {REINFORCE_START.total_reward_lst_trn}')

