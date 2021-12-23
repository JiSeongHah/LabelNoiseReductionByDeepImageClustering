import torch
import numpy as np

def mk_noisy_data(raw_data,noise_ratio,split_ratio,way='sum'):
    if way == 'sum':
        data_with_noise = np.vstack((raw_data[:split_ratio],np.clip(0, 255, (raw_data[split_ratio:] +
            noise_ratio * np.random.randint(0, 255, size=(len(raw_data[split_ratio:]), 28, 28) )  ) )     ) ).astype(int)
    if way == 'pureonly':
        data_with_noise = raw_data[:split_ratio]

    return data_with_noise



# x = np.ones((500,28,28))
#
# y =  mk_noisy_data(raw_data=x,noise_ratio=1.23,split_ratio=28,way='sum')
#
# print(y[29,:,:])