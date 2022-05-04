import torch
from torch.optim import AdamW, Adam
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models
from CALAVG import cal_avg_error
from MY_MODELS import BasicBlock, ResNet, ArcMarginProduct
from save_funcs import mk_name,lst2csv
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import time

class innerTestPredictor(nn.Module):
    def __init__(self,
                 plotSaveDir,
                 ModelName,
                 bSizeTrn=32,
                 bSizeVal=32,
                 gpuUse=True,
                 iterToAccumul=2,
                 beta4f1=100):
        super(innerPredictor, self).__init__()


        self.beta4f1 = beta4f1
        self.num4epoch = 0

        self.plotSaveDir = plotSaveDir
        self.ModelName = ModelName

        self.total_reward_lst = []

        self.bSizeTrn = bSizeTrn
        self.bSizeVal = bSizeVal

        self.gpuUse = gpuUse
        self.MaxStepTrn = MaxStepTrn
        self.MaxStepVal = MaxStepVal
        self.iterToAccumul = iterToAccumul

        if self.gpuUse == True:
            USE_CUDA = torch.cuda.is_available()
            print(USE_CUDA)
            self.device = torch.device('cuda' if USE_CUDA else 'cpu')
            print('학습을 진행하는 기기:', self.device)
        else:
            self.device = torch.device('cpu')
            print('학습을 진행하는 기기:', self.device)

        self.MyBackbone = ResNet(block=BasicBlock,
                                 num_blocks=[2, 2, 2, 2],
                                 num_classes=2,
                                 mnst_ver=True)

        self.optimizer = Adam([{'params':self.MyBackbone.parameters()},
                              {'params':self.MyArc.parameters()}],
                              lr=3e-4,  # 학습률
                              eps = 1e-9
                                # 0으로 나누는 것을 방지하기 위한 epsilon 값
                              )



        self.loss_lst_trn = []

        self.b_size_lst_trn_TRUE_POSITIVE = []
        self.b_size_lst_trn_TRUE_NEGATIVE = []
        self.b_size_lst_trn_FALSE_POSITIVE = []
        self.b_size_lst_trn_FALSE_NEGATIVE = []
        self.b_size_lst_trn_POSITIVE = []
        self.b_size_lst_trn_NEGATIVE = []

        self.loss_lst_val = []

        self.b_size_lst_val_TRUE_POSITIVE = []
        self.b_size_lst_val_TRUE_NEGATIVE = []
        self.b_size_lst_val_FALSE_POSITIVE = []
        self.b_size_lst_val_FALSE_NEGATIVE = []
        self.b_size_lst_val_POSITIVE = []
        self.b_size_lst_val_NEGATIVE = []

        self.avg_loss_lst_trn = []
        self.avg_acc_lst_trn_TRUE_POSITIVE = []
        self.avg_acc_lst_trn_TRUE_NEGATIVE = []
        self.avg_acc_lst_trn_FALSE_POSITIVE = []
        self.avg_acc_lst_trn_FALSE_NEGATIVE = []

        self.avg_acc_lst_trn_f1score = []
        self.avg_acc_lst_trn_PRECISION = []
        self.avg_acc_lst_trn_RECALL = []

        self.avg_loss_lst_val = []

        self.avg_acc_lst_val_TRUE_POSITIVE = []
        self.avg_acc_lst_val_TRUE_NEGATIVE = []
        self.avg_acc_lst_val_FALSE_POSITIVE = []
        self.avg_acc_lst_val_FALSE_NEGATIVE = []

        self.avg_acc_lst_val_f1score = []
        self.avg_acc_lst_val_PRECISION = []
        self.avg_acc_lst_val_RECALL = []




    def setDataLoader(self,trnInput,trnLabel,valInput,valLabel):

        self.trnInput = trnInput
        self.trnLabel = trnLabel
        self.valInput = valInput
        self.valLabel = valLabel

        train_data = TensorDataset(self.trnInput, self.trnLabel)
        self.train_dataloader = DataLoader(train_data,
                                           batch_size=self.bSizeTrn,
                                           shuffle=True,
                                           num_workers=2)

        val_data = TensorDataset(self.valInput, self.valLabel)
        self.val_dataloader = DataLoader(val_data,
                                         batch_size=self.bSizeVal,
                                         shuffle=True,
                                         num_workers=2)


    def forward(self, x):
        output = self.MyBackbone(x.float())

        return output

    def calLoss(self,pred,label):

        correct = torch.argmax(pred, axis=1)
        label_pos = label[label == 0]
        label_neg = label[label != 0]

        label_pos_mask = label == 0
        label_neg_mask = label != 0

        correct_mask = correct == label
        mixed_mask_pos = label_pos_mask * correct_mask
        mixed_mask_neg = label_neg_mask * correct_mask

        b_size_true_pos = len(correct[mixed_mask_pos])
        b_size_true_neg = len(correct[mixed_mask_neg])
        b_size_false_pos = len(label_neg) - b_size_true_neg
        b_size_false_neg = len(label_pos) - b_size_true_pos

        crossentropy_loss = nn.CrossEntropyLoss()
        cross_loss = crossentropy_loss(pred, label)

        b_size_total = len(label)
        b_size_pos = len(label_pos)
        b_size_neg = len(label_neg)

        return cross_loss, b_size_true_pos,b_size_true_neg,b_size_false_pos,b_size_false_neg,b_size_total,b_size_pos,b_size_neg

    def flushLst(self):

        self.loss_lst_trn = []

        self.b_size_lst_trn_TRUE_POSITIVE = []
        self.b_size_lst_trn_TRUE_NEGATIVE = []
        self.b_size_lst_trn_FALSE_POSITIVE = []
        self.b_size_lst_trn_FALSE_NEGATIVE = []
        self.b_size_lst_trn_POSITIVE = []
        self.b_size_lst_trn_NEGATIVE = []


        self.loss_lst_val = []

        self.b_size_lst_val_TRUE_POSITIVE = []
        self.b_size_lst_val_TRUE_NEGATIVE = []
        self.b_size_lst_val_FALSE_POSITIVE = []
        self.b_size_lst_val_FALSE_NEGATIVE = []
        self.b_size_lst_val_POSITIVE = []
        self.b_size_lst_val_NEGATIVE = []

        print('flushing lst done on inner model level')


    def trainingStep(self):

        self.MyBackbone.to(self.device)
        self.MyBackbone.train()

        TDataLoader = tqdm(self.train_dataloader)

        self.optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            globalTime= time.time()

            for idx, (bInput, bLabel)  in enumerate(TDataLoader):


                localTime= time.time()
                bInput = bInput.to(self.device)
                bLabel = bLabel.to(self.device)

                bLogit = self.forward(bInput)
                bLogit = bLogit.cpu()
                bLabel = bLabel.cpu()

                cross_loss,\
                b_size_true_pos,\
                b_size_true_neg,\
                b_size_false_pos,\
                b_size_false_neg,\
                b_size_total,\
                b_size_pos,\
                b_size_neg \
                    = self.calLoss(pred=bLogit, label=bLabel)


                ResultLoss = cross_loss/self.iterToAccumul
                ResultLoss.backward()

                self.loss_lst_trn.append(ResultLoss.clone().detach().item())
                self.b_size_lst_trn_TRUE_POSITIVE.append(b_size_true_pos)
                self.b_size_lst_trn_TRUE_NEGATIVE.append(b_size_true_neg)
                self.b_size_lst_trn_FALSE_POSITIVE.append(b_size_false_pos)
                self.b_size_lst_trn_FALSE_NEGATIVE.append(b_size_false_neg)
                self.b_size_lst_trn_POSITIVE.append(b_size_pos)
                self.b_size_lst_trn_NEGATIVE.append(b_size_neg)

                if (idx + 1) % self.iterToAccumul == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if idx == self.MaxStepTrn:
                    break


                localTimeElaps = round(time.time() - localTime,2)
                globalTimeElaps = round(time.time() - globalTime,2)


                TDataLoader.set_description(f'Processing : {idx}')
                TDataLoader.set_postfix(Gelapsed=globalTimeElaps,Lelapsed=localTimeElaps,loss=ResultLoss.item())


        torch.set_grad_enabled(False)
        self.MyBackbone.eval()
        self.MyBackbone.to('cpu')


    def validatingStep(self):
        self.MyBackbone.to(self.device)
        self.MyBackbone.eval()
        self.optimizer.zero_grad()

        VDataLoader= tqdm(self.val_dataloader)

        with torch.set_grad_enabled(False):
            for idx, (valBInput, valBLabel) in enumerate(VDataLoader):

                valBInput = valBInput.to(self.device)
                valBLabel = valBLabel.to(self.device)

                valBLogit = self.forward(valBInput)
                valBLogit = valBLogit.cpu()
                valBLabel = valBLabel.cpu()

                ResultLoss,\
                b_size_true_pos,\
                b_size_true_neg,\
                b_size_false_pos,\
                b_size_false_neg,\
                b_size_total,\
                b_size_pos,\
                b_size_neg \
                    = self.calLoss(pred=valBLogit, label=valBLabel)

                self.loss_lst_val.append(float(ResultLoss.clone().detach().item()))
                self.b_size_lst_val_TRUE_POSITIVE.append(b_size_true_pos)
                self.b_size_lst_val_TRUE_NEGATIVE.append(b_size_true_neg)
                self.b_size_lst_val_FALSE_POSITIVE.append(b_size_false_pos)
                self.b_size_lst_val_FALSE_NEGATIVE.append(b_size_false_neg)
                self.b_size_lst_val_POSITIVE.append(b_size_pos)
                self.b_size_lst_val_NEGATIVE.append(b_size_neg)


                VDataLoader.set_description(f'Processing : {idx}')
                VDataLoader.set_postfix(loss=ResultLoss)


                if idx == self.MaxStepVal:
                    break


        torch.set_grad_enabled(True)
        self.MyBackbone.train()
        self.MyBackbone.to('cpu')


    def validatingStepEnd(self):

        self.avg_loss_lst_trn.append(np.mean(self.loss_lst_trn*self.iterToAccumul))
        self.avg_loss_lst_val.append(np.mean(self.loss_lst_val))

        TP_trn = np.sum(self.b_size_lst_trn_TRUE_POSITIVE)
        TN_trn = np.sum(self.b_size_lst_trn_TRUE_NEGATIVE)
        FP_trn = np.sum(self.b_size_lst_trn_FALSE_POSITIVE)
        FN_trn = np.sum(self.b_size_lst_trn_FALSE_NEGATIVE)

        PRECISION_trn = TP_trn/(TP_trn+FP_trn+1e-9)
        RECALL_trn = TP_trn/(TP_trn+FN_trn+1e-9)
        F1_score_trn = ((self.beta4f1*self.beta4f1 + 1)*PRECISION_trn*RECALL_trn)/(self.beta4f1*self.beta4f1*PRECISION_trn + RECALL_trn )

        self.avg_acc_lst_trn_TRUE_POSITIVE.append(TP_trn)
        self.avg_acc_lst_trn_TRUE_NEGATIVE.append(TN_trn)
        self.avg_acc_lst_trn_FALSE_POSITIVE.append(FP_trn)
        self.avg_acc_lst_trn_FALSE_NEGATIVE.append(FN_trn)

        self.avg_acc_lst_trn_PRECISION.append(PRECISION_trn)
        self.avg_acc_lst_trn_RECALL.append(RECALL_trn)
        self.avg_acc_lst_trn_f1score.append(F1_score_trn)

        TP_val = np.sum(self.b_size_lst_val_TRUE_POSITIVE)
        TN_val = np.sum(self.b_size_lst_val_TRUE_NEGATIVE)
        FP_val = np.sum(self.b_size_lst_val_FALSE_POSITIVE)
        FN_val = np.sum(self.b_size_lst_val_FALSE_NEGATIVE)


        PRECISION_val = TP_val / (TP_val + FP_val+1e-9)
        RECALL_val = TP_val / (TP_val + FN_val+1e-9)
        F1_score_val = ((self.beta4f1 * self.beta4f1 + 1) * PRECISION_val * RECALL_val) / (
                    self.beta4f1 * self.beta4f1 * PRECISION_val + RECALL_val+1e-9)

        print(f' TP trn is : {TP_trn}, TN trn is : {TN_trn}, FP trn is : {FP_trn}, FN trn is : {FN_trn}')
        print(f'PRECISION trn : {PRECISION_trn}, RECALL trn : {RECALL_trn}, F1 SCORE trn : {F1_score_trn}')
        print(f' TP val is : {TP_val}, TN val is : {TN_val}, FP val is : {FP_val}, FN val is : {FN_val}')
        print(f'PRECISION VAL : {PRECISION_val}, RECALL VAL : {RECALL_val}, F1 SCORE VAL : {F1_score_val}')

        self.avg_acc_lst_val_TRUE_POSITIVE.append(TP_val)
        self.avg_acc_lst_val_TRUE_NEGATIVE.append(TN_val)
        self.avg_acc_lst_val_FALSE_POSITIVE.append(FP_val)
        self.avg_acc_lst_val_FALSE_NEGATIVE.append(FN_val)

        self.avg_acc_lst_val_PRECISION.append(PRECISION_val)
        self.avg_acc_lst_val_RECALL.append(RECALL_val)
        self.avg_acc_lst_val_f1score.append(F1_score_val)

        fig = plt.figure(constrained_layout=True)
        ax1 = fig.add_subplot(2, 4, 1)
        ax1.plot(range(len(self.avg_loss_lst_trn)), self.avg_loss_lst_trn)
        ax1.set_title('train loss')
        ax2 = fig.add_subplot(2, 4, 2)
        ax2.plot(range(len(self.avg_acc_lst_trn_PRECISION)), self.avg_acc_lst_trn_PRECISION)
        ax2.set_title('train PRECISION')
        ax3 = fig.add_subplot(2, 4, 3)
        ax3.plot(range(len(self.avg_acc_lst_trn_RECALL)), self.avg_acc_lst_trn_RECALL)
        ax3.set_title('train RECALL')
        ax4 = fig.add_subplot(2, 4, 4)
        ax4.plot(range(len(self.avg_acc_lst_trn_f1score)), self.avg_acc_lst_trn_f1score)
        ax4.set_title('train F1 SCORE')

        ax5 = fig.add_subplot(2, 4, 5)
        ax5.plot(range(len(self.avg_loss_lst_val)), self.avg_loss_lst_val)
        ax5.set_title('val loss')
        ax6 = fig.add_subplot(2, 4, 6)
        ax6.plot(range(len(self.avg_acc_lst_val_PRECISION)), self.avg_acc_lst_val_PRECISION)
        ax6.set_title('val PRECISION')
        ax7 = fig.add_subplot(2, 4, 7)
        ax7.plot(range(len(self.avg_acc_lst_val_RECALL)), self.avg_acc_lst_val_RECALL)
        ax7.set_title('val RECALL')
        ax8 = fig.add_subplot(2, 4, 8)
        ax8.plot(range(len(self.avg_acc_lst_val_f1score)), self.avg_acc_lst_val_f1score)
        ax8.set_title('val F1 SCORE')

        plt.savefig(self.plotSaveDir + str(self.ModelName)+'inner_model_result.png', dpi=200)
        print('saving inner plot complete!')
        plt.close()

        self.flushLst()

    def FIT(self,iterationNum,stopThreshold):

        for i in range(iterationNum):
            self.trainingStep()
            self.validatingStep()
            self.validatingStepEnd()

            if len(self.avg_acc_lst_val_f1score) > 12:

                avg_error =cal_avg_error(x=self.avg_acc_lst_val_f1score[-10:],y=self.avg_acc_lst_val_f1score[-11:-1])

                if avg_error <stopThreshold:
                    print(f'avg_error : {avg_error} is smaller than threshold : {stopThreshold} so break now')
                    break
                else:
                    print(f'avg_error : {avg_error} is bigger than threshold : {stopThreshold} so keep iterating')



    def VALIDATE(self):

        self.validatingStep()

    def GETSCORE(self):

        return torch.mean(self.avg_acc_lst_val_f1score[-10:])















