import os
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from pathlib import Path
import pandas as pd
from PIL import Image

class CSVDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.targets = self.data['label'].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        image = Image.open(item['impath'])
        label = item['label']

        if image.mode == 'L':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    data_mode = getattr(args, "data_mode", "image_folder")

    if data_mode == "image_folder":
        root = os.path.join(args.data_path, is_train)
        dataset = datasets.ImageFolder(root, transform=transform)
    else:
        csv_path = getattr(args, "csv_path", None)
        assert csv_path is not None, "csv_path should be provided for csv data_mode"
        root = os.path.join(args.data_path, f"{Path(csv_path).stem}_{is_train}.csv")
        dataset = CSVDataset(root, transform=transform)

    if is_train == 'train':
        ratio = float(getattr(args, "dataratio", 1.0))
        seed = int(getattr(args, "seed", 0))
        stratified = bool(getattr(args, "stratified", False))

        if 0.0 < ratio < 1.0:
            if stratified:
                idx = _stratified_indices(dataset.targets, ratio, seed)
            else:
                # simple uniform subsample with torch.Generator for reproducibility
                g = torch.Generator().manual_seed(seed)
                n = len(dataset)
                k = max(1, int(n * ratio))
                idx = torch.randperm(n, generator=g)[:k].tolist()
            dataset = Subset(dataset, idx)

    return dataset

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    if is_train == 'train':
        return create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )

    # eval transform
    crop_pct = 224 / 256 if args.input_size <= 224 else 1.0
    size = int(args.input_size / crop_pct)
    t = [
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(t)

# ---- helpers ----

def _stratified_indices(targets, ratio: float, seed: int):
    """Maintain class proportions. Ensures at least 1 sample per class when possible."""
    t = torch.as_tensor(targets)
    classes = torch.unique(t)
    g = torch.Generator().manual_seed(seed)

    keep = []
    for c in classes.tolist():
        cls_idx = torch.nonzero(t == c, as_tuple=False).view(-1)
        if len(cls_idx) == 0:
            continue
        k = max(1, int(round(len(cls_idx) * ratio)))
        sel = cls_idx[torch.randperm(len(cls_idx), generator=g)[:k]]
        keep.extend(sel.tolist())

    # shuffle final indices (stable across seed)
    g2 = torch.Generator().manual_seed(seed + 1)
    keep = torch.tensor(keep)[torch.randperm(len(keep), generator=g2)].tolist()
    return keep

