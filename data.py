import logging
from pathlib import Path

import lightning as L
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from embeddings import generate_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AcouslicAIDataModule(L.LightningDataModule):
    def __init__(
        self,
        task: str,
        data_dir: str,
        num_workers: int,
        few_shot_list: None | list = None,
        batch_size: int = 32,
        image_transform=None,
        mask_transform=None,
        use_augmentation: bool = True,
    ):
        super().__init__()
        assert task in ["classification", "segmentation"], "Invalid task type"
        self.task = task
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.few_shot_list = few_shot_list
        self.use_augmentation = use_augmentation

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.embedding_paths_dict = None

        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def prepare_data(self):
        # please either download the raw dataset from https://zenodo.org/records/12697994
        # or use the preprocessed dataset using the link on our github repository
        ...

    def prepare_embeddings(self, model, num_workers: int = 12, num_batches: int = 128):
        if self.embedding_paths_dict is not None:
            return

        dataloader = DataLoader(
                ConcatDataset(
                    [self.train_dataset, self.val_dataset, self.test_dataset]
                ),
                batch_size=num_batches,
                num_workers=num_workers,
                pin_memory=True,
            )

        generate_embeddings(model, dataloader, task=self.task)
        
        self.embedding_paths_dict = (
            self.train_dataset.embedding_paths_dict
            | self.val_dataset.embedding_paths_dict
            | self.test_dataset.embedding_paths_dict
        )

    def get_embedding_paths_dict(self):
        embedding_paths_dict = {}
        for dataset in [
            self.train_dataset.data,
            self.val_dataset.data,
            self.test_dataset,
        ]:
            for data_path, emb_path in dataset.data:
                key = data_path.stem
                embedding_paths_dict[key] = emb_path
        return embedding_paths_dict

    def setup(self, stage: str):
        if stage == "embeddings":
            self.train_dataset = EmbeddingDataset(
                task=self.task,
                data_dir=self.data_dir / "train" / "data",
                image_transform=self.image_transform,
                few_shot_list=self.few_shot_list,
                use_augmentation=self.use_augmentation,
            )
            self.val_dataset = EmbeddingDataset(
                task=self.task,
                data_dir=self.data_dir / "val" / "data",
                image_transform=self.image_transform,
                few_shot_list=self.few_shot_list,
                use_augmentation=self.use_augmentation,
            )
            self.test_dataset = EmbeddingDataset(
                task=self.task,
                data_dir=self.data_dir / "test" / "data",
                image_transform=self.image_transform,
                few_shot_list=self.few_shot_list,
                use_augmentation=self.use_augmentation,
            )
        elif stage == "fit":
            self.train_dataset = AcoudslicAIDataset(
                data_dir=self.data_dir / "train" / "data",
                image_transform=self.image_transform,
                mask_transform=self.mask_transform,
                few_shot_list=self.few_shot_list,
                use_augmentation=self.use_augmentation,
                embedding_paths_dict=self.embedding_paths_dict,
            )
            self.val_dataset = AcoudslicAIDataset(
                data_dir=self.data_dir / "val" / "data",
                image_transform=self.image_transform,
                mask_transform=self.mask_transform,
                few_shot_list=self.few_shot_list,
                use_augmentation=self.use_augmentation,
                embedding_paths_dict=self.embedding_paths_dict,
            )
            self.test_dataset = AcoudslicAIDataset(
                data_dir=self.data_dir / "test" / "data",
                image_transform=self.image_transform,
                mask_transform=self.mask_transform,
                few_shot_list=self.few_shot_list,
                use_augmentation=self.use_augmentation,
                embedding_paths_dict=self.embedding_paths_dict,
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


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        task: str,
        data_dir: str | Path,
        image_transform=None,
        few_shot_list=None,
        use_augmentation: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.image_transform = image_transform

        self.embedding_paths_dict = {}
        self.unprocessed_data = []
        for data_path in sorted(self.data_dir.glob("*.npz")):
            if not use_augmentation and len(data_path.stem.split("_")) >= 3:
                continue
            if few_shot_list is not None and data_path.stem not in few_shot_list:
                continue

            emb_path = (
                data_path.parent.parent / "embeddings" / task / f"{data_path.stem}.pt"
            )
            self.embedding_paths_dict[data_path.stem] = emb_path
            if not emb_path.exists():
                emb_path.parent.mkdir(parents=True, exist_ok=True)
                self.unprocessed_data.append((data_path, emb_path))

    def __len__(self):
        return len(self.unprocessed_data)

    def __getitem__(self, idx):
        data = self.unprocessed_data[idx]
        image_mask = np.load(data[0])

        image = image_mask["image"]

        if self.image_transform:
            image = Image.fromarray(image)
            image = self.image_transform(image)

        return {"image": image, "emb_path": str(data[1])}


class AcoudslicAIDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        image_transform=None,
        mask_transform=None,
        few_shot_list=None,
        use_augmentation: bool = True,
        embedding_paths_dict=None,
    ):
        self.data_dir = Path(data_dir)
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.embedding_paths_dict = embedding_paths_dict

        self.data = []
        for data_path in sorted(self.data_dir.glob("*.npz")):
            if not use_augmentation and len(data_path.stem.split("_")) >= 3:
                continue
            if few_shot_list is not None and data_path.stem not in few_shot_list:
                continue

            if self.embedding_paths_dict is not None:
                self.data.append((data_path, self.embedding_paths_dict[data_path.stem]))
            else:
                self.data.append((data_path,))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        image_mask = np.load(data[0])

        image = image_mask["image"]
        mask = image_mask["mask"]
        # 1 indicates presence of the mask, 0 indicates absence
        label = (mask.sum() > 0).astype(np.float32)[None]

        if self.image_transform:
            image = Image.fromarray(image)
            image = self.image_transform(image)
        if self.mask_transform:
            mask = Image.fromarray(mask)
            mask = self.mask_transform(mask)

        item = {"image": image, "mask": mask, "label": label}
        if self.embedding_paths_dict is not None:
            embs = torch.load(data[1])
            item["embs"] = embs
        return item
