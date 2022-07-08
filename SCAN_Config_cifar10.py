
dataConfigs = dict(
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

