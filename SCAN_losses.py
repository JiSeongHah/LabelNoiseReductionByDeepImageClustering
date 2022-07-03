import torch.nn as nn
import numpy as np
import torch

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