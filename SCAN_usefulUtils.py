import numpy as np
from scipy.stats import mode
import pickle


def getMinHeadIdx(loadDir):

    with open(loadDir+'minLossHeadIdx.pkl','rb') as f:
        minLossHeadIdxLst = pickle.load(f)

    modeIdx = mode(minLossHeadIdxLst[-10:]).mode[0]

    return modeIdx

def getAccPerConfLst(Dict,linspaceNum,minConf=0.95):

    confLst = []
    for i in Dict.keys():
        confLst.append(i)

    print(f'len of confLst is : {len(confLst)}')
    print(f'mean of confLst is : {np.mean(confLst)}')
    print(f'there is {confLst.count(1.0)} of 1 in confLst ')

    if minConf is None:
        minConf = np.min(confLst)
    maxConf = np.max(confLst)
    print(f'max of confLst is : {maxConf}')

    xLst = np.linspace(minConf,maxConf,linspaceNum)

    totalXDict = {}
    for rangeNum in range(len(xLst)-1):
        totalXDict[str(round(xLst[rangeNum],2))+'~'+str(round(xLst[rangeNum+1],2))] = []
    totalXDict['under_'+str(round(xLst[0],2))] = []
    totalXDict['over_'+str(round(xLst[-1],2))] = []

    for eachConf,eachResult in Dict.items():
        if eachConf < xLst[0]:
            totalXDict['under_' + str(round(xLst[0], 2))].append(eachResult)
        if eachConf >= xLst[-1]:
            totalXDict['over_' + str(round(xLst[-1], 2))].append(eachResult)
        for rangeNum in range(len(xLst)-1):
            if xLst[rangeNum] <=eachConf < xLst[rangeNum+1]:
                # print(f'yes because eachConf : {eachConf} is smalller than {xLst[rangeNum+1]}')
                totalXDict[str(round(xLst[rangeNum], 2)) + '~' + str(round(xLst[rangeNum + 1], 2))].append(
                    eachResult)
            else:
                pass

    finalConf =[]
    finalAcc = []
    finalAllocNum = []

    tt = 0
    for key,values in totalXDict.items():
        print(f'{len(values)} numbers allocated for key :{key}')
        tt += len(values)
    print(f'len of data is : {tt}')

    finalConf.append('under_'+str(round(xLst[0],2)))
    finalAcc.append(np.mean(totalXDict['under_'+str(round(xLst[0],2))]))
    finalAllocNum.append(len(totalXDict['under_'+str(round(xLst[0],2))]))
    for rangeNum in range(len(xLst)-1):
        finalConf.append(str(round(xLst[rangeNum], 2)) + '~' + str(round(xLst[rangeNum + 1], 2)))
        finalAcc.append(np.mean(totalXDict[str(round(xLst[rangeNum], 2)) + '~' + str(round(xLst[rangeNum + 1], 2))]))
        finalAllocNum.append(len(totalXDict[str(round(xLst[rangeNum], 2)) + '~' + str(round(xLst[rangeNum + 1], 2))]))
    finalConf.append('over_' + str(round(xLst[-1], 2)))
    finalAcc.append(np.mean(totalXDict['over_' + str(round(xLst[-1], 2))]))
    finalAllocNum.append(len(totalXDict['over_' + str(round(xLst[-1], 2))]))


    return finalConf, finalAcc,finalAllocNum












