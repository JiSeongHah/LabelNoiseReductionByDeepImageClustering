import torch
import numpy as np

def getData(Dir):

    RL_train_dataset = MNIST(Dir, train=True, download=True)
    RL_val_dataset = MNIST(Dir, train=False, download=True)

    RL_train_data = RL_train_dataset.data.numpy()
    RL_train_label = RL_train_dataset.targets.numpy()

    RL_train_label_zero = RL_train_label[RL_train_label == 0]
    RL_train_label_rest = RL_train_label[RL_train_label != 0]
    RL_train_label_rest = torch.ones_like(RL_train_label_rest)

    RL_train_data_zero = RL_train_data[RL_train_label == 0]
    RL_train_data_rest = RL_train_data[RL_train_label != 0]

    RL_val_inputs = torch.from_numpy(RL_val_dataset.data.numpy()).clone().detach().unsqueeze(1)
    RL_val_labels = torch.from_numpy(RL_val_dataset.targets.numpy()).clone().detach()
    RL_val_labels = torch.ones_like(RL_val_labels)
    

    if self.wayofdata == 'sum':
        RL_train_data_zero_little = torch.from_numpy(
            mk_noisy_data(raw_data=RL_train_data_zero, noise_ratio=self.noise_ratio,
                          split_ratio=self.split_ratio, way=self.wayofdata)).unsqueeze(1)
        RL_train_label_zero_little = torch.from_numpy(RL_train_label_zero[:])
    elif self.wayofdata == 'pureonly':
        RL_train_data_zero_little = torch.from_numpy(
            mk_noisy_data(raw_data=RL_train_data_zero, noise_ratio=self.noise_ratio,
                          split_ratio=self.split_ratio, way=self.wayofdata)).unsqueeze(1)
        RL_train_label_zero_little = torch.from_numpy(RL_train_label_zero[:self.split_ratio])
    elif self.wayofdata == 'noiseonly':
        RL_train_data_zero_little = torch.from_numpy(
            mk_noisy_data(raw_data=RL_train_data_zero, noise_ratio=self.noise_ratio,
                          split_ratio=self.split_ratio, way=self.wayofdata)).unsqueeze(1)
        RL_train_label_zero_little = torch.from_numpy(RL_train_label_zero[:])


    totalTrainInput = torch.cat((RL_train_data_rest,RL_train_data_zero_little),dim=0)
    totalTrainLabel = torch.cat((RL_train_label_rest,RL_train_label_zero_little),dim=0)

    return totalTrainInput,totalTrainLabel, RL_val_inputs,RL_val_labels