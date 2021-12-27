import torch
from torch.optim import AdamW, Adam
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models
from CALAVG import cal_avg_error
from MY_MODELS import BasicBlock, ResNet
from save_funcs import mk_name,lst2csv
import matplotlib.pyplot as plt

class Prediction_lit_4REINFORCE1(pl.LightningModule):
    def __init__(self,save_dir,save_range,stop_threshold,beta4f1=100):
        super().__init__()

        # self.model = CNN()
        # self.model = models.resnet18(pretrained=False)
        #
        # self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # #Change the output layer to output 10 classes instead of 1000 classes
        # num_ftrs = self.model.fc.in_features
        # self.model.fc = nn.Linear(num_ftrs, 2)
        self.model = ResNet(block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=2, mnst_ver=True)

        self.save_dir = save_dir

        self.stop_threshold = stop_threshold
        self.beta4f1 = beta4f1

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

        self.num4epoch = 0
        self.save_range = save_range
        self.total_reward_lst = []


    def flush_lst(self):

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

        print('flushing lst done on model level')

    def forward(self, x):
        output = self.model(x.float())

        return output

    def crossentropy_loss(self, pred, label):

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

    def training_step(self, train_batch, batch_idx):
        b_input, b_label = train_batch
        logits = self(b_input)

        cross_loss, b_size_true_pos, b_size_true_neg, b_size_false_pos, b_size_false_neg, b_size_total, b_size_pos, b_size_neg\
            = self.crossentropy_loss(pred=logits, label=b_label)

        self.loss_lst_trn.append(float(cross_loss.clone().detach().item()))
        self.b_size_lst_trn_TRUE_POSITIVE.append(b_size_true_pos)
        self.b_size_lst_trn_TRUE_NEGATIVE.append(b_size_true_neg)
        self.b_size_lst_trn_FALSE_POSITIVE.append(b_size_false_pos)
        self.b_size_lst_trn_FALSE_NEGATIVE.append(b_size_false_neg)
        self.b_size_lst_trn_POSITIVE.append(b_size_pos)
        self.b_size_lst_trn_NEGATIVE.append(b_size_neg)


        return cross_loss

    def validation_step(self, val_batch, batch_idx):
        val_b_input, val_b_label = val_batch
        logits = self(val_b_input)
        #print('11111111111111111111111111111111111111111111111111111111111111111111')
        cross_loss, b_size_true_pos, b_size_true_neg, b_size_false_pos, b_size_false_neg, b_size_total, b_size_pos, b_size_neg\
            = self.crossentropy_loss(pred=logits, label=val_b_label)

        #print(f'ajoijweoifjaoiwejfio is : {cross_loss}')

        self.loss_lst_val.append(float(cross_loss.clone().detach().item()))

        self.b_size_lst_val_TRUE_POSITIVE.append(b_size_true_pos)
        self.b_size_lst_val_TRUE_NEGATIVE.append(b_size_true_neg)
        self.b_size_lst_val_FALSE_POSITIVE.append(b_size_false_pos)
        self.b_size_lst_val_FALSE_NEGATIVE.append(b_size_false_neg)
        self.b_size_lst_val_POSITIVE.append(b_size_pos)
        self.b_size_lst_val_NEGATIVE.append(b_size_neg)

        return cross_loss

    def validation_epoch_end(self,validiation_step_outputs):

        self.avg_loss_lst_trn.append(np.mean(self.loss_lst_trn))
        self.avg_loss_lst_val.append(np.mean(self.loss_lst_val))

        TP_trn = np.sum(self.b_size_lst_trn_TRUE_POSITIVE)
        TN_trn = np.sum(self.b_size_lst_trn_TRUE_NEGATIVE)
        FP_trn = np.sum(self.b_size_lst_trn_FALSE_POSITIVE)
        FN_trn = np.sum(self.b_size_lst_trn_FALSE_NEGATIVE)

        PRECISION_trn = TP_trn/(TP_trn+FP_trn)
        RECALL_trn = TP_trn/(TP_trn+FN_trn)
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

        fig = plt.figure()
        ax1 = fig.add_subplot(2, 4, 1)
        ax1.plot(range(len(self.avg_loss_lst_trn)), self.avg_loss_lst_trn)
        ax1.set_title('train loss')
        ax2 = fig.add_subplot(2, 4, 2)
        ax2.plot(range(len(self.avg_acc_lst_trn_PRECISION)), self.avg_acc_lst_trn_PRECISION)
        ax2.set_title('train PRECISION')
        ax3 = fig.add_subplot(2, 4, 3)
        ax3.plot(range(len(self.avg_acc_lst_trn_RECALL)),self.avg_acc_lst_trn_RECALL)
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

        plt.savefig(self.save_dir+'.png', dpi=400)
        print('saving plot complete!')
        plt.close()


        # if len(self.avg_loss_lst_val)>6:
        #     print(f'avg error of last 5 loss_val is : {cal_avg_error(self.loss_lst_val[-5:], self.loss_lst_val[-6:-1])} while stop_threshold : {self.stop_threshold} ')
        #
        # if len(self.avg_acc_lst_val_f1score)>6:
        #     print(f'avg error of last 5 f1score val is : {cal_avg_error(self.avg_acc_lst_val_f1score[-5:], self.avg_acc_lst_val_f1score[-6:-1])} while stop_threshold : {self.stop_threshold} ')


        # if self.num4epoch % self.save_range == 0:
        #
        #     saving_name = mk_name(self.num4epoch,ls_val=round(self.avg_loss_lst_val[-1],2),PREC=round(self.avg_acc_lst_val_PRECISION[-1],2),
        #                           REC=round(self.avg_acc_lst_val_RECALL[-1],2),F1SCRE=round(self.avg_acc_lst_val_f1score[-1],2))
        #
        #     #print(self.avg_acc_lst_total_trn,self.avg_acc_lst_zero_trn,self.avg_acc_lst_rest_trn)
        #
        #     lst2csv(save_dir=self.save_dir,save_name=saving_name,loss_trn=list(self.avg_loss_lst_trn),
        #             PRECISION_trn = list(self.avg_acc_lst_trn_PRECISION),RECALL_trn=list(self.avg_acc_lst_trn_RECALL),
        #             F1SCORE_trn = list(self.avg_acc_lst_trn_f1score),loss_val= list(self.avg_loss_lst_val),
        #             PRECISION_val= list(self.avg_acc_lst_val_PRECISION),RECALL_val = list(self.avg_acc_lst_val_RECALL),
        #             F1SCORE_val = list(self.avg_acc_lst_val_f1score))
        #
        #     fig = plt.figure()
        #     ax1 = fig.add_subplot(2, 4, 1)
        #     ax1.plot(range(len(self.avg_loss_lst_trn)), self.avg_loss_lst_trn)
        #     ax1.set_title('train loss')
        #     ax2 = fig.add_subplot(2, 4, 2)
        #     ax2.plot(range(len(self.avg_acc_lst_trn_PRECISION)), self.avg_acc_lst_trn_PRECISION)
        #     ax2.set_title('train PRECISION')
        #     ax3 = fig.add_subplot(2, 4, 3)
        #     ax3.plot(range(len(self.avg_acc_lst_trn_RECALL)),self.avg_acc_lst_trn_RECALL)
        #     ax3.set_title('train RECALL')
        #     ax4 = fig.add_subplot(2, 4, 4)
        #     ax4.plot(range(len(self.avg_acc_lst_trn_f1score)), self.avg_acc_lst_trn_f1score)
        #     ax4.set_title('train F1 SCORE')
        #
        #     ax5 = fig.add_subplot(2, 4, 5)
        #     ax5.plot(range(len(self.avg_loss_lst_val)), self.avg_loss_lst_val)
        #     ax5.set_title('val loss')
        #     ax6 = fig.add_subplot(2, 4, 6)
        #     ax6.plot(range(len(self.avg_acc_lst_val_PRECISION)), self.avg_acc_lst_val_PRECISION)
        #     ax6.set_title('val PRECISION')
        #     ax7 = fig.add_subplot(2, 4, 7)
        #     ax7.plot(range(len(self.avg_acc_lst_val_RECALL)), self.avg_acc_lst_val_RECALL)
        #     ax7.set_title('val RECALL')
        #     ax8 = fig.add_subplot(2, 4, 8)
        #     ax8.plot(range(len(self.avg_acc_lst_val_f1score)), self.avg_acc_lst_val_f1score)
        #     ax8.set_title('val F1 SCORE')
        #
        #     plt.savefig(self.save_dir+saving_name+'.png', dpi=400)
        #     print('saving plot complete!')
        #     plt.close()

        self.num4epoch +=1
        self.flush_lst()

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(),
                          lr=4e-6,  # 학습률
                          eps=1e-9  # 0으로 나누는 것을 방지하기 위한 epsilon 값
                          )
        return optimizer
