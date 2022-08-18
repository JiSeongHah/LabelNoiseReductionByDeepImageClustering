"""
    @inproceedings{vangansbeke2020scan,
    title={Scan: Learning to classify images without labels},
    author={Van Gansbeke, Wouter and Vandenhende, Simon and Georgoulis, Stamatios and Proesmans, Marc and Van Gool, Luc},
    booktitle={Proceedings of the European Conference on Computer Vision},
    year={2020}
}
"""

import torch
import numpy as np
from tqdm import tqdm


# scan train code, code for step B-1
def scanTrain(train_loader,headNum, featureExtractor,ClusterHead, criterion, optimizer,accumulNum, device, update_cluster_head_only=True):
    """
    Train w/ SCAN-Loss
    """

    if update_cluster_head_only:
        featureExtractor.eval()  # No need to update BN
    else:
        featureExtractor.train()  # Update BN

    optimizer[0].zero_grad()
    #0 th optimizer is for feature extraction model
    optimizer[1].zero_grad()
    #1 th optimizer if for cluster head

    # dict contains total loss from each cluster head
    totalLossDict = dict()
    for h in range(headNum):
        totalLossDict[f'head_{h}'] = []

    # dict contains loss (1) only from each cluster head
    consistencyLossDict = dict()
    for h in range(headNum):
        consistencyLossDict[f'head_{h}'] = []

    # dict constains entropy loss only from each cluster head
    entropyLossDict = dict()
    for h in range(headNum):
        entropyLossDict[f'head_{h}'] = []

    with torch.set_grad_enabled(True):
        for i, batch in enumerate(train_loader):
            print(f'{i}/{len(train_loader)} batch training start... ')
            # Forward pass
            # anchors is data which becomes criterion to load neighbor of it
            anchors = batch['anchor'].to(device)
            # neighbor is literally neighbor of data : 'anchors'
            neighbors = batch['neighbor'].to(device)

            if update_cluster_head_only:  # Only calculate gradient for backprop of linear layer

                with torch.no_grad():
                    anchors_features = featureExtractor(anchors)
                    neighbors_features = featureExtractor(neighbors)
                anchors_output = ClusterHead.forward(anchors_features,inputDiff=False)
                neighbors_output = ClusterHead.forward(neighbors_features,inputDiff=False)

            else:  # Calculate gradient for backprop of complete network
                anchors_output = ClusterHead.forward(featureExtractor(anchors),inputDiff=False)
                neighbors_output = ClusterHead.forward(featureExtractor(neighbors),inputDiff=False)

            totalLossInnerDict,consistencyLossInnerDict,entropyLossInnerDict = criterion(anchors_output,neighbors_output)

            totalLoss = sum(loss for loss in totalLossInnerDict.values())

            totalLoss.backward()
            if i%accumulNum == 0:
                if update_cluster_head_only: # only update weight of cluster head only
                    optimizer[1].step()
                    optimizer[1].zero_grad()
                else:
                    # update weight of feature extraction model and cluster head both.
                    optimizer[0].step()
                    optimizer[1].step()
                    optimizer[0].zero_grad()
                    optimizer[1].zero_grad()

            # append each loss to dict
            for h in range(headNum):
                totalLossDict[f'head_{h}'].append(totalLossInnerDict[f'head_{h}'].cpu().item())
                consistencyLossDict[f'head_{h}'].append(consistencyLossInnerDict[f'head_{h}'])
                entropyLossDict[f'head_{h}'].append(entropyLossInnerDict[f'head_{h}'])
            print(f'{i}/{len(train_loader)} batch training complete!!! ')

    return totalLossDict, consistencyLossDict, entropyLossDict


# code for training. with step B-2.
def selflabelTrain(train_loader,headNum, featureExtractor,ClusterHead, criterion, optimizer, device,accumulNum, update_cluster_head_only=True):

    if update_cluster_head_only:
        # because, no gradient for feature extraction model
        # make feature extraction model to eval mode.
        featureExtractor.eval()
    else:
        # because to calculate gradient for feature extraction model.
        # make feature extraction model to train mode.
        featureExtractor.train()  # Update BN

    optimizer[0].zero_grad()
    optimizer[1].zero_grad()
    # gradientStep = len(train_loader)

    totalLossDict = dict()
    for h in range(headNum):
        totalLossDict[f'head_{h}'] = []

    totalLossDict4Plot = dict()
    for h in range(headNum):
        totalLossDict4Plot[f'head_{h}'] = []

    train_loader = tqdm(train_loader)

    for i, batch in enumerate(train_loader):

        images = batch['image'].to(device)
        AugedImage = batch['AugedImage'].to(device)

        if update_cluster_head_only:
            # doesn't calculate gradient for feature extraction model
            with torch.no_grad():
                embed = featureExtractor(images)
                augedEmbed = featureExtractor(AugedImage)

            with torch.no_grad():
                output = ClusterHead.forward(embed,inputDiff=False)
            with torch.set_grad_enabled(True):
                AugedOutput = ClusterHead.forward(augedEmbed,inputDiff=False)
        else:
            # calculate gradient for feature extraction model.
            with torch.no_grad():
                output = ClusterHead.forward(featureExtractor(images),inputDiff=False)
            with torch.set_grad_enabled(True):
                AugedOutput = ClusterHead.forward(featureExtractor(AugedImage),inputDiff=False)

        totalLossInnerDict = criterion(output,AugedOutput)

        finalLoss = sum(loss for loss in totalLossInnerDict.values())
        train_loader.set_description(f'training {i}/{len(train_loader)}')
        train_loader.set_postfix({'loss : ':finalLoss.item()})

        finalLoss.backward()

        if i%accumulNum == 0:
            if update_cluster_head_only:
                optimizer[1].step()
                optimizer[1].zero_grad()
            else:
                optimizer[0].step()
                optimizer[1].step()
                optimizer[0].zero_grad()
                optimizer[1].zero_grad()

        for h in range(headNum):
            totalLossDict4Plot[f'head_{h}'].append(totalLossInnerDict[f'head_{h}'].cpu().item())


    return totalLossDict4Plot



# code for training filtered data, if necessary
# this code helps to train model with high confident dat only
def trainWithFiltered(train_loader, featureExtractor, ClusterHead, criterion, optimizer, device, accumulNum):

    featureExtractor.train()  # Update BN

    optimizer[0].zero_grad()
    optimizer[1].zero_grad()
    # gradientStep = len(train_loader)

    totalLossLst = []
    totalAccLst = []

    train_loader = tqdm(train_loader)
    with torch.set_grad_enabled(True):
        for i, batch in enumerate(train_loader):

            images = batch['image'].to(device)
            labels = batch['label']


            output = ClusterHead.forward(featureExtractor(images)).cpu()

            totalLoss ,acc = criterion(output,labels)

            train_loader.set_description(f'training {i}/{len(train_loader)}')
            train_loader.set_postfix({'loss : ': totalLoss.item()})

            totalLoss.backward()

            if i % accumulNum == 0:
                optimizer[0].step()
                optimizer[1].step()
                optimizer[0].zero_grad()
                optimizer[1].zero_grad()

            totalLossLst.append(totalLoss.item())
            totalAccLst.append(acc)

    return totalLossLst, totalAccLst




































































































