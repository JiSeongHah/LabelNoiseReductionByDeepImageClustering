import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F


class MaskedCELoss(nn.Module):
    def __init__(self):
        super(MaskedCELoss, self).__init__()

    def forward(self, input, label, mask, weight, reduction='mean'):
        if not (mask != 0).any():
            raise ValueError('Mask in MaskedCrossEntropyLoss is all zeros.')
        label = torch.masked_select(label, mask)
        b, c = input.size()
        n = label.size(0)
        input = torch.masked_select(input, mask.view(b, 1)).view(n, c)
        return F.cross_entropy(input, label, weight=weight, reduction=reduction)


class ConfidenceBasedCE(nn.Module):
    def __init__(self, threshold, weight4CE=True):
        super(ConfidenceBasedCE, self).__init__()
        self.loss = MaskedCELoss()
        self.softmax = nn.Softmax(dim=1)
        self.threshold = threshold
        self.weight4CE = weight4CE

    def forward(self, anchorsAugWeak, anchorsAugStrong):
        """
        Loss function during self-labeling
        input: logits for original samples and for its strong augmentations
        output: cross entropy
        """
        # Retrieve target and mask based on weakly augmentated anchors
        weakAnchorsProb = self.softmax(anchorsAugWeak)
        maxProbs, targetIndices = torch.max(weakAnchorsProb, dim=1)
        mask = maxProbs > self.threshold
        b, c = weakAnchorsProb.size()
        targetIndicesMasked = torch.masked_select(targetIndices, mask.squeeze())
        n = targetIndicesMasked.size(0)

        # Inputs are strongly augmented anchors
        input_ = anchorsAugStrong

        # Class balancing weights
        if self.weight4CE:
            idx, counts = torch.unique(targetIndicesMasked, return_counts=True)
            freq = 1 / (counts.float() / n)
            weight = torch.ones(c).cuda()
            weight[idx] = freq

        else:
            weight = None

        # Loss
        loss = self.loss(input_, targetIndicesMasked, mask, weight=weight, reduction='mean')

        return loss


def entropy(x, input_as_probabilities=True):
    """
    Helper function to compute the entropy over the batch
    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ = torch.clamp(x, min=1e-8)
        b = x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)

    if len(b.size()) == 2:  # Sample-wise entropy
        return -b.sum(dim=1).mean()
    elif len(b.size()) == 1:  # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' % (len(b.size())))


class SCANLoss(nn.Module):
    def __init__(self, entropyWeight=2.0):
        super(SCANLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.bce = nn.BCELoss()
        self.entropyWeight = entropyWeight  # Default = 2.0

    def forward(self, anchors, neighbors):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]
        output:
            - Loss
        """
        # Softmax

        headNum = len(anchors.keys())

        totalLossDict = dict()
        consistencyLossDict = dict()
        entropyLossDict = dict()

        for h in range(headNum):
            anchorsProbPerHead = self.softmax(anchors[f'eachHead_{h}'])
            positivesProbPerHead = self.softmax(neighbors[f'eachHead_{h}'])

            b, n = anchorsProbPerHead.size()

            # Similarity in output space
            similarity = torch.bmm(anchorsProbPerHead.view(b, 1, n), positivesProbPerHead.view(b, n, 1)).squeeze()
            ones = torch.ones_like(similarity)
            consistencyLoss = self.bce(similarity, ones)

            # Entropy loss
            entropyLoss = entropy(torch.mean(anchorsProbPerHead, dim=0), input_as_probabilities=True)

            # Total loss
            totalLoss = consistencyLoss - self.entropyWeight * entropyLoss

            totalLossDict[f'head_{h}'] = totalLoss
            consistencyLossDict[f'head_{h}'] = consistencyLoss.item()
            entropyLossDict[f'head_{h}'] = entropyLoss.item()


        return totalLossDict, consistencyLossDict, entropyLossDict