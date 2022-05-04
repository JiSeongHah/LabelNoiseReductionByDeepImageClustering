import torch
import numpy as np
from torchvision.datasets import MNIST

def func_getData(Dir):
    #############################LOAD DATA##################
    RL_train_dataset = MNIST(Dir, train=True, download=True)
    RL_val_dataset = MNIST(Dir, train=False, download=True)
    #############################LOAD DATA##################

    ###########     TRAIN DATA PART #######################
    RL_train_data = torch.from_numpy(RL_train_dataset.data.numpy()).unsqueeze(1)
    RL_train_label = RL_train_dataset.targets.numpy()

    RL_train_label_zero = RL_train_label[RL_train_label == 0]
    RL_train_label_rest = RL_train_label[RL_train_label != 0]
    RL_train_label_rest = torch.ones_like(torch.from_numpy(RL_train_label_rest))
    RL_train_label_zero = torch.from_numpy(RL_train_label_zero)

    RL_train_data_zero = RL_train_data[RL_train_label == 0]
    RL_train_data_rest = RL_train_data[RL_train_label != 0]
    ###########     TRAIN DATA PART #######################




    ###########    VAL DATA PART ##########################
    RL_val_inputs = torch.from_numpy(RL_val_dataset.data.numpy()).clone().detach().unsqueeze(1)
    RL_val_labels = torch.from_numpy(RL_val_dataset.targets.numpy()).clone().detach()

    RL_val_labels_zero = RL_val_labels[RL_val_labels == 0]
    RL_val_labels_rest = RL_val_labels[RL_val_labels != 0]
    RL_val_labels_rest = torch.ones_like(RL_val_labels_rest)

    RL_val_data_zero = RL_val_inputs[RL_val_labels == 0]
    RL_val_data_rest = RL_val_inputs[RL_val_labels != 0]
    ###########    VAL DATA PART ##########################




    return RL_train_data_rest,\
           RL_train_label_rest,\
           RL_train_data_zero,\
           RL_train_label_zero,\
           RL_val_data_rest,\
           RL_val_labels_rest,\
           RL_val_data_zero,\
           RL_val_labels_zero