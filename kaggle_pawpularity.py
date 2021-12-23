# import torch
# from torch.utils.data import Dataset
# import os
# from glob import glob
# from PIL import Image
# import numpy as np
# import pandas as pd
# from torchvision import transforms
# from torch.utils.data import DataLoader
# import torch
# import json
# from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# from torch.optim import AdamW, SGD
# import matplotlib.pyplot as plt
# import copy
# import pandas as pd
# import numpy as np
# import random
# import time
# import datetime
# import os
# import argparse
# import torch.nn.functional as F
# import torch.nn as nn
# from torch.nn import CrossEntropyLoss
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import Callback
# import multiprocessing
# from glob import glob
# from torch.distributions import Categorical
# from torchvision import models
# from torchvision.datasets import MNIST
# from torch.nn import MSELoss
# import csv
# from converge_test import conv_test
#
# class CustomDataset(Dataset):
#     def __init__(self, data_dir, label_dir):
#         self.data_dir = data_dir
#         self.label_dir = label_dir
#
#         self.data_lst = glob(data_dir + '*')
#
#         self.label_raw = pd.read_csv(self.label_dir).to_numpy()
#         self.label_names = self.label_raw[:, 0]
#         self.label_pawps = self.label_raw[:, -1]
#
#         self.pawp_dict = dict()
#
#         print('starting making dict')
#
#         for name, pawp in zip(self.label_names, self.label_pawps):
#             self.pawp_dict[str(name)] = float(pawp)
#
#         print('making dict complete')
#
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0, 0, 0), (0.5, 0.5, 0.5)),
#             transforms.Resize([900, 800])
#         ])
#
#     def __len__(self):
#         return len(self.data_lst)
#
#     def __getitem__(self, idx):
#         image = self.transform(Image.open(self.data_lst[idx]))
#
#         file_name = str(self.data_lst[idx]).split('/')[-1].split('.')[0]
#         label = float(self.pawp_dict[file_name])
#
#         return image, label
#
#
# # config 클래스
# class Config(dict):
#     __getattr__ = dict.__getitem__
#     __setattr__ = dict.__setattr__
#
#     @classmethod
#     def load(cls, file):
#         with open(file, 'r') as f:
#             config = json.loads(f.read())
#             return Config(config)
#
#
# manager = multiprocessing.Manager()
#
#
# class Pawpularity(pl.LightningModule):
#     def __init__(self, lr=4e-6,conv_threshold=0.1,range_num=5):
#         super().__init__()
#
#         self.model = models.resnet18(pretrained=True)
#
#         # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#
#         # Change the output layer to output 10 classes instead of 1000 classes
#         num_ftrs = self.model.fc.in_features
#         self.model.fc = nn.Linear(num_ftrs, 1)
#
#         self.MSE_loss_lst = []
#         self.avg_MSE_loss_lst = []
#
#         self.MSE_loss_lst_val = []
#         self.avg_MSE_loss_lst_val = []
#
#         self.lr = lr
#
#         self.conv_threshold = conv_threshold
#         self.range_num =range_num
#
#     def flush_lst(self):
#         self.MSE_loss_lst = []
#         self.MSE_loss_lst_val = []
#
#         print('flushing lst done on pl model level')
#
#     def forward(self, x):
#         result = self.model(x.float())
#         return result
#
#     def MSE_LOSS(self, pred_pawp, label):
#         which_loss = MSELoss()
#
#         loss_output = which_loss(pred_pawp, label)
#         return loss_output
#
#     def training_step(self, train_batch, batch_idx):
#         # print('training start......')
#         b_input, b_label = train_batch
#
#         b_label = b_label.float()
#
#         pred_pawp = self.forward(b_input)
#
#         total_loss = self.MSE_LOSS(pred_pawp, b_label)
#
#         self.MSE_loss_lst.append(float(total_loss.clone().detach()))
#         return total_loss
#
#     #     def training_epoch_end(self, training_step_outputs):
#     #         print('optimizing start...........')
#
#     #         print('optimizing complete!!!')
#
#     def validation_step(self, val_batch, batch_idx):
#         val_b_input, val_b_label = val_batch
#         val_b_label = val_b_label.float()
#         val_pred_pawp = self(val_b_input)
#
#         val_total_loss = self.MSE_LOSS(val_pred_pawp, val_b_label)
#         self.MSE_loss_lst_val.append(float(val_total_loss.clone().detach()))
#
#         return val_total_loss
#
#     def validation_epoch_end(self, validation_step_outputs):
#         print('validation_epoch_end start...........')
#
#         self.avg_MSE_loss_lst.append(np.mean(self.MSE_loss_lst))
#         self.avg_MSE_loss_lst_val.append(np.mean(self.MSE_loss_lst_val))
#
#         try:
#
#
#             fig = plt.figure()
#
#             ax1 = fig.add_subplot(1, 2, 1)
#             ax1.plot(range(len(self.avg_MSE_loss_lst)), self.avg_MSE_loss_lst)
#             ax1.set_title('train MSE loss')
#
#             ax2 = fig.add_subplot(1, 2, 2)
#             ax2.plot(range(len(self.avg_MSE_loss_lst_val)), self.avg_MSE_loss_lst_val)
#             ax2.set_title('validation MSE loss')
#
#             plt.show()
#             plt.close()
#
#         except:
#             print('somthing wrong')
#
#
#
#         self.flush_lst()
#         print('validation_epoch_end complete!!!')
#
#     def configure_optimizers(self):
#         optimizer = SGD(self.model.parameters(),
#                         lr=self.lr  # 학습률
#                         # 0으로 나누는 것을 방지하기 위한 epsilon 값
#                         )
#         return optimizer
#
#
# class datamodule(pl.LightningDataModule):
#     def __init__(self, train_data_dir, train_label_dir, val_data_dir, val_label_dir, batch_size=10, batch_size_val=10):
#         super().__init__()
#
#         self.batch_size = batch_size
#         self.batch_size_val = batch_size_val
#         self.number_of_epoch = 0
#
#         print('theta setup start....')
#
#         self.download_root = './'
#
#         self.train_data_dir = train_data_dir
#         self.train_label_dir = train_label_dir
#         self.val_data_dir = val_data_dir
#         self.val_label_dir = val_label_dir
#
#     def prepare_data(self, stage=None):
#         pass
#
#     def flush_data(self):
#         self.train_inputs = 0
#         self.train_labels = 0
#         self.val_inputs = 0
#         self.val_labels = 0
#         pass
#         # print('flushing data done.')
#
#     def setup(self, stage=None):
#
#         if stage == 'fit' or stage is None:
#             print('theta stage is ', stage)
#             self.train_dataset = CustomDataset(data_dir=self.train_data_dir, label_dir=self.train_label_dir)
#             self.val_dataset = CustomDataset(data_dir=self.train_data_dir, label_dir=self.val_label_dir)
#             pass
#
#         if stage == 'test' or stage is None:
#             pass
#
#     def train_dataloader(self):
#         print('train_dataloading.......')
#
#         train_sampler = RandomSampler(self.train_dataset)
#         train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.batch_size,
#                                       num_workers=8)
#
#         print('train_dataloading done....')
#
#         return train_dataloader
#
#     def val_dataloader(self):
#
#         validation_sampler = SequentialSampler(self.val_dataset)
#         validation_dataloader = DataLoader(self.val_dataset, sampler=validation_sampler, batch_size=self.batch_size_val,
#                                            num_workers=8)
#         return validation_dataloader
#
#     def test_dataloader(self):
#         pass
#         # test_data = TensorDataset(self.test_inputs, self.test_labels)
#         # test_sampler = RandomSampler(test_data)
#         # test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.batch_size, num_workers=4)
#
#         # return test_dataloader
#
#
# if __name__ == '__main__':
#     train_bs =16
#     val_bs = 16
#     learning_rate = 4e-6
#
#     train_data_dir = '/home/emeraldsword1423/Downloads/petfinder/train/'
#     train_label_dir = '/home/emeraldsword1423/Downloads/petfinder/train.csv'
#     val_data_dir = '/home/emeraldsword1423/Downloads/petfinder/test/'
#     val_label_dir = '/home/emeraldsword1423/Downloads/petfinder/test.csv'
#
#     model = Pawpularity(lr=learning_rate)
#     dm = datamodule(batch_size=train_bs, batch_size_val=val_bs,
#                     train_data_dir=train_data_dir, train_label_dir=train_label_dir,
#                     val_data_dir=train_data_dir, val_label_dir=train_label_dir)
#
#     trainer = pl.Trainer(gpus=1, accelerator='dp',
#                          logger=False, num_sanity_val_steps=0, weights_summary=None)
#
#     trainer.fit(model, dm)
#
import pandas

if __name__ == '__main__':
    sample = pandas.read_csv('/home/emeraldsword1423/Downloads/petfinder/sample_submission.csv')
    test_result = sample.to_csv('/home/emeraldsword1423/Downloads/petfinder/submission1.csv',index=False)
