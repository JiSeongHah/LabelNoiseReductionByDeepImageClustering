import torch
from mnst_arcface_innerPredictor import innerPredictor


def func_LoadInnerPredictor(modelSaveLoadDir,
                            innermodelLoadNum,
                            plotSaveDir,
                            bSizeTrn,
                            bSizeVal,
                            LinNum,
                            s,
                            m,
                            gpuUse,
                            MaxStepTrn,
                            MaxStepVal,
                            iterToAccumul,
                            beta4f1=100
                            ):

    try:
        LoadedModel = torch.load(modelSaveLoadDir+innermodelLoadNum+'.pth')
        print('loading model successed!!!')
    except:
        print('loading model failed so load fresh model')
        LoadedModel = innerPredictor(
            plotSaveDir = plotSaveDir,
            bSizeTrn = bSizeTrn,
            bSizeVal = bSizeVal,
            LinNum = LinNum,
            s = s,
            m = m,
            gpuUse = gpuUse,
            MaxStepTrn = MaxStepTrn,
            MaxStepVal = MaxStepVal,
            iterToAccumul = iterToAccumul,
            beta4f1= beta4f1
        )

    return LoadedModel
