
dataConfigs = dict(
    type="cifar10",
    trans1=dict(
        aug_type="weak",
        crop_size=32,
        normalize=dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
    ),
    trans2=dict(
        aug_type="scan",
        crop_size=32,
        normalize=dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
        num_strong_augs=4,
        cutout_kwargs=dict(n_holes=1,
                           length=16,
                           random=True)
    ),
)

