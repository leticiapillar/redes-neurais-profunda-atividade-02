"""Dataset PyTorch para o LGG Brain MRI Segmentation (imagem FLAIR/pré/pós-contraste
+ máscara binária de anormalidade)."""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class BrainMRISegmentationDataset(Dataset):
    def __init__(
        self,
        manifest_df,
        image_size: int = 128,
        augment: bool = False,
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
    ):
        self.df = manifest_df.reset_index(drop=True)
        self.image_size = image_size
        self.augment = augment
        self.mean = torch.tensor(mean).view(3, 1, 1) if mean is not None else None
        self.std = torch.tensor(std).view(3, 1, 1) if std is not None else None

    def __len__(self):
        return len(self.df)

    def _load_pair(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(PROJECT_ROOT / row.image_path).convert("RGB")
        mask = Image.open(PROJECT_ROOT / row.mask_path).convert("L")
        return image, mask

    def __getitem__(self, idx):
        image, mask = self._load_pair(idx)

        image = TF.resize(image, [self.image_size, self.image_size])
        mask = TF.resize(
            mask, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST
        )

        if self.augment:
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            if random.random() < 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        image = TF.to_tensor(image)  # [0, 1], shape (3, H, W)
        if self.mean is not None:
            image = (image - self.mean) / self.std

        mask = torch.from_numpy((np.array(mask) > 127).astype(np.float32)).unsqueeze(0)

        return image, mask
