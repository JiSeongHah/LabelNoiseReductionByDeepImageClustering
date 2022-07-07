import torch
import numpy as np


def scanTrain(train_loader,headNum, featureExtractor,ClusterHead, criterion, optimizer, device, update_cluster_head_only=True):
    """
    Train w/ SCAN-Loss
    """

    if update_cluster_head_only:
        featureExtractor.eval()  # No need to update BN
    else:
        featureExtractor.train()  # Update BN

    optimizer.zero_grad()
    # gradientStep = len(train_loader)

    totalLossDict = dict()
    for h in range(headNum):
        totalLossDict[f'head_{h}'] = []

    consistencyLossDict = dict()
    for h in range(headNum):
        consistencyLossDict[f'head_{h}'] = []

    entropyLossDict = dict()
    for h in range(headNum):
        entropyLossDict[f'head_{h}'] = []

    with torch.set_grad_enabled(True):
        for i, batch in enumerate(train_loader):
            print(f'{i}/{len(train_loader)} batch training start... ')
            # Forward pass
            anchors = batch['anchor'].to(device)
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
            optimizer.zero_grad()
            totalLoss.backward()
            optimizer.step()

            for h in range(headNum):
                totalLossDict[f'head_{h}'].append(totalLossInnerDict[f'head_{h}'].cpu().item())
                consistencyLossDict[f'head_{h}'].append(consistencyLossInnerDict[f'head_{h}'])
                entropyLossDict[f'head_{h}'].append(entropyLossInnerDict[f'head_{h}'])
            print(f'{i}/{len(train_loader)} batch training complete!!! ')

    return totalLossDict, consistencyLossDict, entropyLossDict

def selflabelTrain(train_loader, model, criterion, optimizer, epoch, ema=None):
    """
    Self-labeling based on confident samples
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), [losses],
                             prefix="Epoch: [{}]".format(epoch))
    model.train()

    for i, batch in enumerate(train_loader):
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)

        with torch.no_grad():
            output = model(images)[0]
        output_augmented = model(images_augmented)[0]

        loss = criterion(output, output_augmented)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema is not None:  # Apply EMA to update the weights of the network
            ema.update_params(model)
            ema.apply_shadow(model)

