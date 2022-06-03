from mnst_arcface_outer import OuterLoop
import os
from save_funcs import mk_name,createDirectory
import matplotlib.pyplot as plt
import csv

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
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
threshold = 0.85
splitRatioLst = [i/10 for i in range(1,10)]
noiseRatioLst = [i/5 for i in range(1,16)]




for splitRatio in splitRatioLst:
    accCompareLst = []
    accBaselineLst = []

    for noiseRatio in noiseRatioLst:
        print(f'start {noiseRatio} th experiment')
        specificName = mk_name(dirFeatureVer='/',
                               s=s,
                               m=m,
                               threshold = threshold,
                               splitRatio = splitRatio,
                               noiseRatio=noiseRatio,
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

        accCompare,accBaseline = doit.executeFeatureNoiseVer1(usePretrained=False,
                                                                       threshold=threshold,
                                                                       splitRatio= splitRatio,
                                                                       noiseRatio=noiseRatio,
                                                                       iterNum=1000)

        accCompareLst.append(accCompare)
        accBaselineLst.append(accBaseline)
        idxLst = [i for i in range(len(accCompareLst))]
        totalResult = [accCompareLst, accBaselineLst, idxLst]
        with open(plotSaveDir + 'ReusltCSVVer.csv', 'w') as F:
            wr = csv.writer(F)
            wr.writerows(totalResult)
        print(f'{noiseRatio} th experiment complete')

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
    plt.plot(range(len(noiseRatioLst)),accBaselineLst)
    plt.plot(range(len(noiseRatioLst)), accCompareLst)
    plt.savefig(dataDir+'Result/'+str(threshold)+'_'+str(splitRatio)+'_FinalAcc.png',dpi=200)