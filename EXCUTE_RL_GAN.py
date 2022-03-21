import torch
from torchvision.datasets import MNIST
from MK_NOISED_DATA import mk_noisy_data
from REINFORCE_GAN_TORCH import REINFORCE_GAN_TORCH
from save_funcs import load_my_model ,createDirectory
import numpy as np


class EXCUTE_RL_GAN():
    def __init__(self,
                 gamma,
                 eps,
                 theta_b_size,
                 theta_stop_threshold,
                 trn_fle_down_path,
                 theta_gpu_num,
                 theta_max_epch,
                 wayofdata,
                 noise_ratio,
                 split_ratio,
                 RL_save_range,
                 dNoise,
                 dHidden,
                 rl_lr,
                 rl_b_size,
                 gan_trn_bSize,
                 gan_val_bSize,
                 reward_normalize,
                 rwd_spread,
                 beta4f1,
                 rl_stop_threshold,
                 test_fle_down_path,
                 model_save_load_path,
                 max_ep,
                 max_step_trn,
                 max_step_val,
                 reward_method,
                 whichGanLoss,
                 GLoadNum,
                 GbaseLoadNum,
                 DLoadNum,
                 DVRLLoadNum,
                 ):

        ####################################VARS FOR CLASS : REINFORCE_TORCH ############################
        self.rl_b_size = rl_b_size
        self.theta_b_size = theta_b_size
        self.gamma = gamma
        self.rl_lr = rl_lr
        self.reward_normalize = reward_normalize

        self.test_fle_down_path = test_fle_down_path
        self.model_save_load_path = model_save_load_path
        self.model_save_load_pathG = self.model_save_load_path + 'MODEL_GAN_G/'
        createDirectory(self.model_save_load_pathG)
        self.model_save_load_pathD = self.model_save_load_path + 'MODEL_GAN_D/'
        createDirectory(self.model_save_load_pathD)
        self.model_save_load_pathGbase = self.model_save_load_path + 'MODEL_GAN_GBASE/'
        createDirectory(self.model_save_load_pathGbase)
        self.model_save_load_pathDVRL = self.model_save_load_path + 'MODEL_DVRL/'
        createDirectory(self.model_save_load_pathDVRL)
        self.GLoadNum = GLoadNum
        self.GbaseLoadNum = GbaseLoadNum
        self.DLoadNum = DLoadNum
        self.DvrlLoadNum = DVRLLoadNum

        self.theta_gpu_num = theta_gpu_num

        self.MAX_EP = max_ep
        self.theta_max_epch = theta_max_epch
        self.theta_stop_threshold = theta_stop_threshold
        self.rl_stop_threshold = rl_stop_threshold
        self.rwd_spread = rwd_spread
        self.beta4f1 = beta4f1

        self.RL_save_range = RL_save_range
        self.eps = eps
        self.reward_method = reward_method

        self.max_step_trn = max_step_trn
        self.max_step_val = max_step_val
        self.gan_trn_bSize = gan_trn_bSize
        self.gan_val_bSize = gan_val_bSize
        self.dNoise = dNoise
        self.dHidden = dHidden
        self.whichGanLoss = whichGanLoss

        ####################################VARS FOR CLASS : REINFORCE_TORCH ############################
        self.noise_ratio = noise_ratio
        self.split_ratio = split_ratio
        self.wayofdata = wayofdata
        self.trn_fle_down_path = trn_fle_down_path
        ####################################VARS FOR CLASS : EXCUTE_RL ############################


    def excute_RL(self,**vars):

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
            REINFORCE_START.updateVars(**vars)
            REINFORCE_START.loadSavedModel()
            print(f'REINFORCE_START.model_num_now is : {REINFORCE_START.model_num_now}')
            print(f'len of REINFORCE_START.total_reward_lst_trn : {len(REINFORCE_START.total_reward_lst_trn)}')

            print('model loading doneeeeeeeeeeeeee')
            #time.sleep(5)
            print('successsuccesssuccesssuccesssuccesssuccesssuccesssuccess')
        except:
            print('model loading failed so loaded fresh model')
            REINFORCE_START = REINFORCE_GAN_TORCH(gamma=self.gamma,
                                                  eps=self.eps,
                                                 dNoise=self.dNoise,
                                                 dHidden=self.dHidden,
                                                 rl_lr=self.rl_lr,
                                                 rl_b_size=self.rl_b_size,
                                                 gan_trn_bSize=self.gan_trn_bSize,
                                                 gan_val_bSize=self.gan_val_bSize,
                                                 reward_normalize=self.reward_normalize,
                                                 val_data=RL_val_inputs,
                                                 val_label=RL_val_labels,
                                                 rwd_spread=self.rwd_spread,
                                                 beta4f1=self.beta4f1,
                                                 rl_stop_threshold=self.rl_stop_threshold,
                                                 test_fle_down_path=self.test_fle_down_path,
                                                 model_save_load_path=self.model_save_load_path,
                                                 model_save_load_pathG = self.model_save_load_pathG,
                                                 model_save_load_pathGbase = self.model_save_load_pathGbase,
                                                 model_save_load_pathD = self.model_save_load_pathD,
                                                 model_save_load_pathDVRL = self.model_save_load_pathDVRL,
                                                 max_step_trn=self.max_step_trn,
                                                 max_step_val=self.max_step_val,
                                                 reward_method=self.reward_method,
                                                 whichGanLoss=self.whichGanLoss,
                                                 GLoadNum = self.GLoadNum,
                                                 GbaseLoadNum = self.GbaseLoadNum,
                                                 DLoadNum = self.DLoadNum,
                                                 DvrlLoadNum = self.DvrlLoadNum
                                                  )

            print('failedfailedfailedfailedfailedfailedfailedfailedfailedfailed')
            REINFORCE_START.updateVars(**vars)


        for i in range(10000):
            print(f'{i} th training RL start')

            REINFORCE_START.STARTTRNANDVAL(data=RL_train_data_zero_little,label=RL_train_label_zero_little)
            if i%self.RL_save_range ==0 and i!=0:
                try:
                    print(f'REINFORCE_START.model_num_now is : {REINFORCE_START.model_num_now}')
                    torch.save(REINFORCE_START,self.model_save_load_path+str(i+REINFORCE_START.model_num_now)+'.pt')
                    torch.save(REINFORCE_START.REINFORCE_GAN_G.state_dict(),self.model_save_load_pathG+str(i+self.GLoadNum)+'.pt')
                    torch.save(REINFORCE_START.REINFORCE_GAN_D.state_dict(), self.model_save_load_pathD+ str(i+self.DLoadNum) + '.pt')
                    torch.save(REINFORCE_START.REINFORCE_GAN_GBASE.state_dict(), self.model_save_load_pathGbase+ str(i+self.GbaseLoadNum) + '.pt')
                    torch.save(REINFORCE_START.REINFORCE_DVRL.state_dict(), self.model_save_load_pathDVRL+ str(i+self.DvrlLoadNum) + '.pt')
                    print('saving RL model complete')
                    print('saving RL model complete')
                    print('saving RL model complete')
                    print('saving RL model complete')
                    print('saving RL model complete')
                    print('saving RL model complete')
                except:
                    print('saving model failed')
            if np.mean(REINFORCE_START.loss_lst_trn[-10:]) < 0.01:
                break
            print(f'{i} th training for RL done')
            print('                   ')
            print('                   ')
            print('                   ')
            print('                   ')
            print('                   ')
            print('                   ')
            print('                   ')
            print('                   ')
            print(f' reward lst is : {REINFORCE_START.total_reward_lst_trn}')

