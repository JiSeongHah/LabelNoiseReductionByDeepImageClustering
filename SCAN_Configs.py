import torchvision.transforms as transforms

dataConfigs_Cifar10 = dict(
    baseTransform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                              std=[0.2023, 0.1994, 0.2010])]),
    type="cifar10",
    trans1=dict(
        aug_type="weak",
        crop_size=32,
        normalize=dict(mean=[0.4914, 0.4822, 0.4465],
                       std=[0.2023, 0.1994, 0.2010]),
    ),
    trans2=dict(
        aug_type="scan",
        crop_size=32,
        normalize=dict(mean=[0.4914, 0.4822, 0.4465],
                       std=[0.2023, 0.1994, 0.2010]),
        num_strong_augs=4,
        cutout_kwargs=dict(n_holes=1,
                           length=16,
                           random=True)
    ),
)

dataConfigs_Cifar100 = dict(
    baseTransform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                              std=[0.2675, 0.2565, 0.2761])]),
    type="cifar100",
    trans1=dict(
        aug_type="weak",
        crop_size=32,
        normalize=dict(mean=[0.5071, 0.4867, 0.4408],
                       std=[0.2675, 0.2565, 0.2761]),
    ),
    trans2=dict(
        aug_type="scan",
        crop_size=32,
        normalize=dict(mean=[0.5071, 0.4867, 0.4408],
                       std=[0.2675, 0.2565, 0.2761]),
        num_strong_augs=4,
        cutout_kwargs=dict(n_holes=1,
                           length=16,
                           random=True)
    ),
)


dataConfigs_Stl10 = dict(
    baseTransform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])]),
    type="stl10",
    trans1=dict(
        aug_type="weak",
        crop_size=96,
        normalize=dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
    ),
    trans2=dict(
        aug_type="scan",
        crop_size=96,
        normalize=dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
        num_strong_augs=4,
        cutout_kwargs=dict(n_holes=1,
                           length=32,
                           random=True)
    ),
)

dataConfigs_Imagenet50 = dict(
    baseTransform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])]),
    type="imagenet50",
    trans1=dict(
        aug_type="weak",
        crop_size=224,
        normalize=dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
    ),
    trans2=dict(
        aug_type="simclr",
        random_resized_crop =
        crop_size=224,
        normalize=dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),

    ),
)