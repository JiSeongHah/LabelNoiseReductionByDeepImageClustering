from mnst_arcface_outer import OuterLoop
import os
from save_funcs import mk_name,createDirectory
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = "3"
innermodelLoadNum = 300
dataDir = '/home/a286/hjs_dir1/DVMETRIC/'
bSizeTrn = 128
bSizeVal = 128
LinNum = 512
saveRange = 30
s = 64
m = 0.5
gpuUse = True
MaxStepTrn = 1000000000000
MaxStepVal = 10000000000000
iterToAccumul = 2
beta4f1 = 1
iterNum = 100
threshold = 0.95
splitRatioLst = [0.8,0.82,0.84,0.86,0.88,0.9,0.92,0.94,0.96,0.98]
splitRatioLst = [0.9,0.92,0.94,0.96,0.98]
splitRatioLst = [i/300 for i in range(270,300)]

accLst = []

for splitRatio in splitRatioLst:

    specificName = mk_name(dir1='/',
                           s=s,
                           m=m,
                           threshold = threshold,
                           splitRatio = splitRatio,
                           beta4f1=beta4f1,
                           bSizeTrn=bSizeTrn)

    modelSaveLoadDir = dataDir +'Result/'+ specificName+ '/Models/'
    plotSaveDir = dataDir +'Result/'+ specificName+ '/PLOTS/'

    createDirectory(modelSaveLoadDir)
    createDirectory(plotSaveDir)

    doit = OuterLoop(
        modelSaveLoadDir = modelSaveLoadDir,
        plotSaveDir=plotSaveDir,
        innermodelLoadNum = innermodelLoadNum,
        dataDir = dataDir,
        bSizeTrn = bSizeTrn,
        bSizeVal = bSizeVal,
        LinNum = LinNum,
        saveRange=saveRange,
        s =s,
        m =m,
        gpuUse = gpuUse,
        MaxStepTrn = MaxStepTrn,
        MaxStepVal = MaxStepVal,
        iterToAccumul = iterToAccumul,
        beta4f1 = beta4f1
    )

    acc,filteredIndex, noisedLabelZero = doit.executeLabelNoiseVer(usePretrained=False,
                                                                   threshold=threshold,
                                                                   splitRatio= splitRatio,
                                                                   k=7,
                                                                   iterNum=100)

    accLst.append(acc)


print('===============================================')
print('===============================================')
print('===============================================')
print('===============================================')
print('===============================================')
print('===============================================')
print('===============================================')
print('===============================================')
print('')
print('')
print('')
print('')
print('')
print('')
print('')
print('')
print('acc is : ',accLst)
plt.plot(splitRatioLst,accLst)
plt.savefig(dataDir+'Result/'+str(threshold)+'_FinalAcc1.png',dpi=200)