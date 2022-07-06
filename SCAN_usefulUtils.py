import numpy as np
from scipy.stats import mode
import pickle


def getMinHeadIdx(loadDir):

    with open(loadDir+'minLossHeadIdx.pkl','rb') as f:
        minLossHeadIdxLst = pickle.load(f)

    modeIdx = mode(minLossHeadIdxLst[-10:]).mode[0]

    return modeIdx

def getAccPerConfLst(Dict,linspaceNum):

    confLst = []
    for i in Dict.keys():
        confLst.append(i)

    minConf = np.min(confLst)
    maxConf = np.max(confLst)

    xLst = np.linspace(minConf,maxConf,linspaceNum)

    totalXDict = {}
    for rangeNum in range(len(xLst)-1):
        totalXDict[str(round(xLst[rangeNum],2))+'~'+str(round(xLst[rangeNum+1],2))] = []

    for eachConf,eachResult in Dict.items():
        for rangeNum in range(len(xLst)-1):
            if xLst[rangeNum] <= eachConf < xLst[rangeNum+1]:
                totalXDict[str(round(xLst[rangeNum],2)) + '~' + str(round(xLst[rangeNum + 1],2))].append(eachResult)
            else:
                pass

    finalConf =[]
    finalAcc = []

    for Confs,Accs in totalXDict.items():
        finalConf.append(Confs)
        finalAcc.append(np.mean(totalXDict[Confs]))

    return finalConf, finalAcc












