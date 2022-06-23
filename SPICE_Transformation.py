import torchvision.transforms
import torchvision.transforms as transforms
from SPICE_augmentation import Augment, Cutout
import torch

# def get_train_transformations(cfg):
#     if cfg['aug_type'] == 'standard':
#         # Standard augmentation strategy
#         return transforms.Compose([
#             transforms.RandomResizedCrop(cfg['random_resized_crop']),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(cfg['normalize'])
#         ])
#
#     elif cfg['aug_type'] == 'test':
#         # Standard augmentation strategy
#         return transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(cfg['normalize'])
#         ])
#     elif cfg['aug_type'] == 'test_resize':
#         # Standard augmentation strategy
#         return transforms.Compose([
#             transforms.Resize([cfg['size'], cfg['size']]),
#             transforms.ToTensor(),
#             transforms.Normalize(cfg['normalize'])
#         ])
#
#     elif cfg['aug_type'] == 'weak':
#         print(cfg['normalize'])
#         return transforms.Compose([
#             transforms.RandomCrop(cfg['crop_size'], padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(cfg['normalize'])
#         ])
#
#     elif cfg['aug_type'] == 'simclr':
#         # Augmentation strategy from the SimCLR paper
#         return transforms.Compose([
#             transforms.RandomResizedCrop(cfg['random_resized_crop']),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomApply([
#                 transforms.ColorJitter(cfg['color_jitter'])
#             ], p=cfg['color_jitter_random_apply']),
#             transforms.RandomGrayscale(cfg['random_grayscale']),
#             transforms.ToTensor(),
#             transforms.Normalize(cfg['normalize'])
#         ])
#
#     elif cfg['aug_type'] == 'scan':
#         # Augmentation strategy from our paper
#         return transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomCrop(cfg['crop_size']),
#             Augment(cfg['num_strong_augs']),
#             transforms.ToTensor(),
#             transforms.Normalize(cfg['normalize']),
#             Cutout(
#                 n_holes=cfg['cutout_kwargs']['n_holes'],
#                 length=cfg['cutout_kwargs']['length'],
#                 random=cfg['cutout_kwargs']['random'])])
#     elif cfg['aug_type'] == 'gatcluster':
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225])
#         to_tensor = transforms.ToTensor()
#         flip = transforms.RandomHorizontalFlip(0.5)
#         affine = transforms.RandomAffine(degrees=10, translate=[0.1, 0.1], scale=[0.8, 1.2], shear=10)
#         color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.2)
#         T = transforms.Compose([flip, color_jitter, affine, to_tensor, normalize])
#         return T
#
#     else:
#         raise ValueError('Invalid augmentation strategy {}'.format(p['augmentation_strategy']))


def get_train_transformations(cfg):
    if cfg.aug_type == 'standard':
        # Standard augmentation strategy
        return transforms.Compose([
            transforms.RandomResizedCrop(**cfg.random_resized_crop),
            transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            transforms.Normalize(**cfg.normalize)
        ])

    elif cfg.aug_type == 'test':
        # Standard augmentation strategy
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**cfg.normalize)
        ])
    elif cfg.aug_type == 'test_resize':
        # Standard augmentation strategy
        return transforms.Compose([
            transforms.Resize([cfg.size, cfg.size]),
            # transforms.ToTensor(),
            transforms.Normalize(**cfg.normalize)
        ])

    elif cfg.aug_type == 'weak':
        return transforms.Compose([
            transforms.RandomCrop(cfg.crop_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**cfg.normalize)
        ])

    elif cfg.aug_type == 'simclr':
        # Augmentation strategy from the SimCLR paper
        return transforms.Compose([
            transforms.RandomResizedCrop(**cfg.random_resized_crop),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(**cfg.color_jitter)
            ], p=cfg.color_jitter_random_apply),
            transforms.RandomGrayscale(**cfg.random_grayscale),
            # transforms.ToTensor(),
            transforms.Normalize(**cfg.normalize)
        ])

    elif cfg.aug_type == 'scan':
        # Augmentation strategy from our paper
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(cfg.crop_size),
            Augment(cfg.num_strong_augs),
            # transforms.ToTensor(),
            transforms.Normalize(**cfg.normalize),
            Cutout(
                n_holes=cfg.cutout_kwargs.n_holes,
                length=cfg.cutout_kwargs.length,
                random=cfg.cutout_kwargs.random)])
    elif cfg.aug_type == 'gatcluster':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        to_tensor = transforms.ToTensor()
        flip = transforms.RandomHorizontalFlip(0.5)
        affine = transforms.RandomAffine(degrees=10, translate=[0.1, 0.1], scale=[0.8, 1.2], shear=10)
        color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.2)
        T = transforms.Compose([flip, color_jitter, affine, to_tensor, normalize])
        return T

    else:
        raise ValueError('Invalid augmentation strategy {}'.format(p['augmentation_strategy']))



# from SPICE_CONFIG import Config
#
#
# from importlib import import_module
# import torch
# from torchvision import datasets
# from PIL import Image
# import matplotlib.pyplot as plt
# import random
#
# dataset = datasets.CIFAR10(root='/home/emeraldsword1423/',train=True,download=True)
#
# cfg1 =Config.fromfile('./prac4.py')
# trans1 = cfg1.data_train.trans1
#
# for x in range(10):
#     i = random.randrange(0,50000-1)
#
#     img = dataset[i][0]
#
#     plt.imshow(img)
#     plt.show()
#     plt.close()
#     print(trans1)
#
#
#     # x = torch.randn(5,3,32,32)
#
#     y = get_train_transformations(trans1)
#
#
#     TRNS = torchvision.transforms.ToPILImage()
#
#     img2 = TRNS(y(img))
#
#     plt.imshow(img2)
#     plt.show()
#     # z = y(x)
#     # print(z.size())
