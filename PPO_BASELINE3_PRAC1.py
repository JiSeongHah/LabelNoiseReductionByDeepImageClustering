import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import random
import random
from torchvision.datasets import MNIST
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

class MnistEnv(gym.Env):
    def __init__(self, dataset,images_per_episode=1, Random=True):
        super().__init__()

        self.action_space = gym.spaces.Discrete(10)
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(1,28, 28),
                                                dtype=np.float32)
        self.step_count = 0

        self.dataset = dataset
        self.Random = Random
        self.dataset_idx = 0

        if self.Random:
            self.idx_lst = list(i for i in range(len(self.dataset)))
            random.shuffle(self.idx_lst)
            self.images_per_episode = len(self.dataset)
        else:
            self.idx_lst = list(i for i in range(len(self.dataset)))
            self.images_per_episode = len(self.dataset)

    def step(self, action):
        done = False
        reward = int(action == self.expected_action)

        self.step_count += 1
        if self.step_count >= self.images_per_episode:
            done = True
            print('1 episode complete')
            obs = 0
        else:
            obs = self._next_obs()



        return obs, reward, done, {}

    def reset(self):
        self.step_count = 0
        self.dataset_idx = 0

        if self.Random:
            self.idx_lst = [i for i in range(len(self.dataset))]
            random.shuffle(self.idx_lst)
        else:
            self.idx_lst = list(i for i in range(len(self.dataset)))

        obs = self._next_obs()

        return obs

    def _next_obs(self):
        if self.Random:
            next_obs_idx = self.idx_lst.pop()
            self.expected_action = int(self.dataset[next_obs_idx][1])
            obs = self.dataset[next_obs_idx][0].numpy()
        else:
            obs = self.dataset[self.dataset_idx][0].numpy()

            self.expected_action = int(self.dataset[self.dataset_idx][1])

            self.dataset_idx += 1
            if self.dataset_idx >= len(self.dataset):
                raise StopIteration()

        return obs


download_root = './'
bs_trn = 1
bs_val = 1

RL_train_dataset = MNIST(download_root, train=True, download=True)
RL_val_dataset = MNIST(download_root, train=False, download=True)

RL_train_data = torch.from_numpy(RL_train_dataset.data.numpy()).clone().detach().unsqueeze(1)
RL_train_label = torch.from_numpy(RL_train_dataset.targets.numpy()).clone().detach()
print(f'shape of train_inputs is : {RL_train_data.shape}')
print('spliting train data done')

print('start val data job....')
RL_val_inputs = torch.from_numpy(RL_val_dataset.data.numpy()).clone().detach().unsqueeze(1)
RL_val_labels = torch.from_numpy(RL_val_dataset.targets.numpy()).clone().detach()
print(f'shape of val_inputs is : {RL_val_inputs.shape}')
print('spliting validation ddddata done')

print('train_dataloading.......')
train_data = TensorDataset(RL_train_data, RL_train_label)
print('train_dataloading done....')
validation_data = TensorDataset(RL_val_inputs, RL_val_labels)



my_env1 = MnistEnv(Random=True,dataset=train_data)
check_env(my_env1)
model = PPO('MlpPolicy',my_env1).learn(total_timesteps=100*len(train_data))

test_env = MnistEnv(Random=True,dataset=validation_data)
obs = test_env.reset()
reward_lst = []
for i in range(len(validation_data)):
    action, _state = model.predict(obs,deterministic=True)
    obs, reward, done, info = test_env.step(action)
    reward_lst.append(reward)
    if done:
        obs = test_env.reset()

print(f'basic result is : {np.mean(reward_lst)}')





