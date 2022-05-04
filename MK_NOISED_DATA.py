import torch
import numpy as np

def mk_noisy_data(raw_data,noise_ratio,split_ratio,way='sum'):
    if way == 'sum':
        data_with_noise = np.vstack((raw_data[:split_ratio],np.clip(0, 255, (raw_data[split_ratio:] +
            noise_ratio * np.random.randint(0, 255, size=(len(raw_data[split_ratio:]), 28, 28) )  ) )     ) ).astype(int)
    if way == 'pureonly':
        data_with_noise = raw_data[:split_ratio]

    return data_with_noise

def mk_noisy_dataTorch(raw_data,noise_ratio,split_ratio,way='sum'):
    if way == 'sum':
        rawPart= raw_data[:split_ratio]
        rawPartwithnoise = torch.clamp(raw_data[split_ratio:]+
                                       (noise_ratio * torch.randint(0,255, (len(raw_data[split_ratio:]),28,28 ) ))
                                       ,min=0,max=255).long()

        # data_with_noise = torch.cat((rawPart,rawPartwithnoise),dim=0)

        return rawPart, rawPartwithnoise

    if way == 'pureonly':
        data_with_noise = raw_data[:split_ratio]

        return data_with_noise

def get_noisy_data(wayofdata,split_ratio,RL_train_data_zero,RL_train_label_zero):


    if wayofdata == 'sum':
        rawPart,rawPartwithNoise = mk_noisy_dataTorch(raw_data=RL_train_data_zero, noise_ratio=self.noise_ratio,
                                                       split_ratio=split_ratio, way=wayofdata)
        labelRaw,labelNoisePart =  RL_train_label_zero[:len(rawPart)],RL_train_label_zero[len(rawPart):]

        return rawPart,rawPartwithNoise,labelRaw,labelNoisePart

    elif wayofdata == 'pureonly':
        RL_train_data_zero_little = mk_noisy_dataTorch(raw_data=RL_train_data_zero, noise_ratio=self.noise_ratio,
                                                       split_ratio=split_ratio, way=wayofdata)
        RL_train_label_zero_little = torch.from_numpy(RL_train_label_zero[:self.split_ratio])

        return RL_train_data_zero_little, RL_train_label_zero_little

    elif wayofdata == 'noiseonly':
        RL_train_data_zero_little = mk_noisy_dataTorch(raw_data=RL_train_data_zero, noise_ratio=self.noise_ratio,
                                                       split_ratio=split_ratio, way=wayofdata)
        RL_train_label_zero_little = torch.from_numpy(RL_train_label_zero[:])


        return RL_train_data_zero_little,RL_train_label_zero_little

def mk_noisy_label(raw_data,raw_label,splitRatio):
    
    rawNum = int(len(raw_label)*splitRatio)
    noiseNum  = len(raw_label)-int(len(raw_label)*splitRatio)

    print(f'rawNum : {rawNum} and noiseNum : {noiseNum}')

    rawSide = torch.zeros(rawNum)
    noiseSide = torch.ones(noiseNum)

    noisedLabels = torch.cat((rawSide,noiseSide))

    return raw_data,noisedLabels



# x = np.ones((500,28,28))
#
# y =  mk_noisy_data(raw_data=x,noise_ratio=1.23,split_ratio=28,way='sum')
#
# print(y[29,:,:])