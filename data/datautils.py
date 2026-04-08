import os
from typing import Tuple
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data.hoi_dataset import BongardDataset
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.fewshot_datasets import *
import data.augmix_ops as augmentations

ID_to_DIRNAME={
    'I': 'ImageNet',
    'A': 'imagenet-a',
    'K': 'ImageNet-Sketch',
    'R': 'imagenet-r',
    'V': 'imagenetv2-matched-frequency-format-val',
    'flower102': 'Flower102',
    'dtd': 'DTD',
    'pets': 'OxfordPets',
    'cars': 'StanfordCars',
    'ucf101': 'UCF101',
    'caltech101': 'Caltech101',
    'food101': 'Food101',
    'sun397': 'SUN397',
    'aircraft': 'fgvc_aircraft',
    'eurosat': 'eurosat'
}


def _list_subdirs(path):
    if not path or not os.path.isdir(path):
        return []
    try:
        return sorted([entry.name for entry in os.scandir(path) if entry.is_dir()])[:30]
    except OSError:
        return []


def _resolve_existing_dir(candidates, set_id, data_root):
    print(f"[data] set={set_id} | data_root={data_root}")
    print("[data] checking candidate directories:")
    for path in candidates:
        print(f"[data]   - {path}")
    for path in candidates:
        if os.path.isdir(path):
            print(f"[data] resolved dataset directory: {path}")
            return path
    available = _list_subdirs(data_root)
    available_str = ", ".join(available) if available else "(none / path does not exist)"
    pretty = "\n".join([f"  - {p}" for p in candidates])
    raise FileNotFoundError(
        f"Could not locate dataset directory for set '{set_id}'.\n"
        f"Given --data root: {data_root}\n"
        f"Tried:\n{pretty}\n"
        f"Available subdirectories under data_root: {available_str}\n"
        "Expected one of these structures:\n"
        "  1) <data_root>/ImageNet/val/<class_folders>\n"
        "  2) <data_root>/val/<class_folders>\n"
        "  3) <data_root>/test/<class_folders>\n"
        "Please pass --data to your real dataset root."
    )


def _looks_like_imagefolder(path):
    if not os.path.isdir(path):
        return False
    try:
        return any(entry.is_dir() for entry in os.scandir(path))
    except OSError:
        return False


def _make_fallback_dataset(transform, set_id):
    # Class counts chosen to stay compatible with expected CLIP classifier outputs.
    fallback_num_classes = {
        'I': 1000,
        'K': 1000,
        'V': 1000,
        'A': 200,
        'R': 200,
    }.get(set_id, 1000)
    print(
        f"[data][fallback] Using torchvision FakeData for set={set_id} "
        f"(size=64, num_classes={fallback_num_classes})."
    )
    return datasets.FakeData(
        size=64,
        image_size=(3, 224, 224),
        num_classes=fallback_num_classes,
        transform=transform,
    )

def build_dataset(set_id, transform, data_root, mode='test', n_shot=None, split="all", bongard_anno=False, allow_fallback=False):
    if set_id == 'I':
        # ImageNet validation set
        candidates = [
            os.path.join(data_root, ID_to_DIRNAME[set_id], "val"),  # data_root/ImageNet/val
            os.path.join(data_root, "val"),                         # data_root/val
            os.path.join(data_root, "test"),                        # data_root/test
            data_root,                                              # data_root is already val directory
        ]
        try:
            testdir = _resolve_existing_dir(candidates, set_id=set_id, data_root=data_root)
            if testdir == data_root and not _looks_like_imagefolder(testdir):
                testdir = _resolve_existing_dir(candidates[:3], set_id=set_id, data_root=data_root)
            testset = datasets.ImageFolder(testdir, transform=transform)
        except FileNotFoundError:
            if not allow_fallback:
                raise
            testset = _make_fallback_dataset(transform, set_id)
    elif set_id in ['A', 'K', 'R', 'V']:
        candidates = [
            os.path.join(data_root, ID_to_DIRNAME[set_id], "images"),  # data_root/<dataset>/images
            os.path.join(data_root, "images"),                         # data_root/images
            os.path.join(data_root, "test"),                           # data_root/test
            data_root,                                                 # data_root is already images directory
        ]
        try:
            testdir = _resolve_existing_dir(candidates, set_id=set_id, data_root=data_root)
            if testdir == data_root and not _looks_like_imagefolder(testdir):
                testdir = _resolve_existing_dir(candidates[:3], set_id=set_id, data_root=data_root)
            testset = datasets.ImageFolder(testdir, transform=transform)
        except FileNotFoundError:
            if not allow_fallback:
                raise
            testset = _make_fallback_dataset(transform, set_id)
    elif set_id in fewshot_datasets:
        if mode == 'train' and n_shot:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode, n_shot=n_shot)
        else:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode)
    elif set_id == 'bongard':
        assert isinstance(transform, Tuple)
        base_transform, query_transform = transform
        testset = BongardDataset(data_root, split, mode, base_transform, query_transform, bongard_anno)
    else:
        raise NotImplementedError
        
    return testset


# AugMix Transforms
def get_preaugment():
    return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])

def augmix(image, preprocess, aug_list, severity=1):
    preaugment = get_preaugment()
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1, use_mta_ops=False):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        self.use_mta_ops = use_mta_ops
        self.mta_ops = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        if self.use_mta_ops:
            views = [self.preprocess(self.mta_ops(x)) for _ in range(self.n_views)]
        else:
            views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        return [image] + views



