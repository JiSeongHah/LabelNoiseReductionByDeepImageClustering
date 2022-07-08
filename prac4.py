from SCAN_CONFIG import Config

configPath = './SCAN_Configs.py'

dataCfg = Config.fromfile(configPath)
cfgScan = dataCfg.dataConfigs_Cifar10
# baseTransform = dataCfg.dataConfigs_Cifar10.baseTransform

print(cfgScan)