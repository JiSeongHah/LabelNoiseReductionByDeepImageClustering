import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pytorch_lightning as pl



class datamodule_4REINFORCE1(pl.LightningDataModule):
    def __init__(self,total_tdata,total_tlabel,val_data,val_label,batch_size=1024,batch_size_val=1024):
        super().__init__()

        self.batch_size = batch_size
        self.batch_size_val = batch_size_val
        self.number_of_epoch = 0

        print('theta setup start....')

        self.train_inputs = total_tdata
        self.train_labels = total_tlabel
        print('theta shape of train input,label is : ', self.train_inputs.size(), self.train_labels.size())
        # for i in self.train_labels:
        #     if i != torch.tensor(0) or i != torch.tensor(1):
        #         print(i)
        print('theta spliting train data done')

        self.val_inputs = val_data
        self.val_labels = val_label

        print(f'shape of val data is : {self.val_inputs.shape}')
        print(f'shape of val label is : {self.val_labels.shape}')
        print('theta spliting validation data done')

    def prepare_data(self, stage=None):
        pass

    def flush_data(self):
        self.train_inputs = 0
        self.train_labels = 0
        self.val_inputs = 0
        self.val_labels = 0
        pass
        #print('flushing data done.')

    def setup(self, stage=None):

        if stage == 'fit' or stage is None:
            print('theta stage is ', stage)

        if stage == 'test' or stage is None:
            pass

    def train_dataloader(self):
        print('train_dataloading.......')
        train_data = TensorDataset(self.train_inputs, self.train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size, num_workers=1)
        print('train_dataloading done....')

        return train_dataloader

    def val_dataloader(self):
        validation_data = TensorDataset(self.val_inputs, self.val_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=self.batch_size_val,
                                           num_workers=1)
        return validation_dataloader

    def test_dataloader(self):
        pass
        # test_data = TensorDataset(self.test_inputs, self.test_labels)
        # test_sampler = RandomSampler(test_data)
        # test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=self.batch_size, num_workers=4)

        # return test_dataloader

