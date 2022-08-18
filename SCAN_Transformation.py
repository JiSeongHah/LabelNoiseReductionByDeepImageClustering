"""
@misc{niu2021spice,
      title={SPICE: Semantic Pseudo-labeling for Image Clustering},
      author={Chuang Niu and Ge Wang},
      year={2021},
      eprint={2103.09382},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
"""


import torchvision.transforms
import torchvision.transforms as transforms
from SCAN_augmentation import Augment, Cutout
import torch



def get_train_transformations(cfg):
    if cfg.aug_type == 'standard':
        # Standard augmentation strategy
        return transforms.Compose([
            transforms.RandomResizedCrop(**cfg.random_resized_crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
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
            transforms.ToTensor(),
            transforms.Normalize(**cfg.normalize)
        ])

    elif cfg.aug_type == 'scan':
        # Augmentation strategy from our paper
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(cfg.crop_size),
            Augment(cfg.num_strong_augs),
            transforms.ToTensor(),
            transforms.Normalize(**cfg.normalize),
            Cutout(
                n_holes=cfg.cutout_kwargs.n_holes,
                length=cfg.cutout_kwargs.length,
                random=cfg.cutout_kwargs.random)])

    else:
        raise ValueError('Invalid augmentation strategy {}'.format(p['augmentation_strategy']))


