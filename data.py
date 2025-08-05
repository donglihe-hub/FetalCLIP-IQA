import logging
from pathlib import Path
from typing import Optional

import lightning as L
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from utils import init_logger

logger = logging.getLogger(__name__)
init_logger()


class AcouslicAIDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        num_workers: int,
        batch_size: int = 32,
        use_augmentation: bool = False,
        few_shot_list: Optional[list] = None,
        image_transform=None,
        mask_transform=None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.use_augmentation = use_augmentation

        self.few_shot_list = few_shot_list

        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # please either download the raw dataset from https://zenodo.org/records/12697994
        # or use the preprocessed dataset using the link on our github repository
        ...

    def setup(self):
        self.train_dataset = AcoudslicAIDataset(
            data_dir=self.data_dir / "train",
            image_transform=self.image_transform,
            mask_transform=self.mask_transform,
            few_shot_list=self.few_shot_list,
            use_augmentation=self.use_augmentation,
        )
        self.val_dataset = AcoudslicAIDataset(
            data_dir=self.data_dir / "val",
            image_transform=self.image_transform,
            mask_transform=self.mask_transform,
            few_shot_list=self.few_shot_list,
            use_augmentation=self.use_augmentation,
        )
        self.test_dataset = AcoudslicAIDataset(
            data_dir=self.data_dir / "test",
            image_transform=self.image_transform,
            mask_transform=self.mask_transform,
            few_shot_list=self.few_shot_list,
            use_augmentation=self.use_augmentation,
        )

        logger.info(f"Train size: {len(self.train_dataset)}")
        logger.info(f"Validation size: {len(self.val_dataset)}")
        logger.info(f"Test size: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class AcoudslicAIDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        image_transform=None,
        mask_transform=None,
        few_shot_list: Optional[list] = None,
        use_augmentation: bool = True,
    ):
        self.data_dir = data_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.data = []
        for data_path in list(sorted(self.data_dir.glob("*.npz"))):
            # using augmented data or not
            if not use_augmentation and len(data_path.stem.split("_")) >= 3:
                continue

            # few-shot learning is not implemented but is possible
            if few_shot_list is not None and data_path.stem not in few_shot_list:
                continue

            self.data.append(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_path = self.data[idx]
        image_mask = np.load(data_path)

        image = image_mask["image"]
        mask = image_mask["mask"]

        # 1 indicates presence of the mask, 0 indicates absence
        label = (mask.max() > 0).astype(np.float32)[None]

        if self.image_transform:
            image = Image.fromarray(image).convert("RGB")
            image = self.image_transform(image)
        if self.mask_transform:
            mask = Image.fromarray(mask)
            mask = self.mask_transform(mask)

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask > 0)
        mask = mask.float()

        item = {"image": image, "mask": mask, "label": label}
        return item
