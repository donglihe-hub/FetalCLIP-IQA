import json
import logging
from pathlib import Path

import Lightning as L
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from embeddings import get_embedding_paths_dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AcouslicAIDataModule(L.LightningDataModule):
    def __init__(
        self,
        task: str,
        data_dir: str,
        num_workers: int,
        few_shot_list: list,
        split_file: Path,
        batch_size: int = 32,
        image_transform=None,
        mask_transform=None,
    ):
        super().__init__()
        assert task in ["classification", "segmentation"], "Invalid task type"
        self.task = task
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.few_shot_list = few_shot_list

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.embedding_dict = None

        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def prepare_data(self):
        # please either download the raw dataset from https://zenodo.org/records/12697994
        # or use the preprocessed dataset using the link on our github repository
        ...

    def prepare_embeddings(self, model):
        if self.embedding_dict is not None:
            return

        self.embedding_dict = get_embedding_paths_dict(
            model,
            ConcatDataset([self.train_dataset, self.val_dataset, self.test_dataset]),
        )
        self.train_dataset.add_embedding_paths_dict(self.embedding_dict)
        self.val_dataset.add_embedding_paths_dict(self.embedding_dict)
        self.test_dataset.add_embedding_paths_dict(self.embedding_dict)

    def setup(self):
        with open(self.split_file, "r") as f:
            # Load the split file which contains the list of few-shot classes
            self.few_shot_list = json.load(f)
        self.train_dataset = AcoudslicAIDataset(
            data_dir=self.data_dir / "train",
            image_transform=self.image_transform,
            mask_transform=self.mask_transform,
            few_shot_list=self.few_shot_list,
        )
        self.val_dataset = AcoudslicAIDataset(
            data_dir=self.data_dir / "val",
            image_transform=self.image_transform,
            mask_transform=self.mask_transform,
            few_shot_list=self.few_shot_list,
        )
        self.test_dataset = AcoudslicAIDataset(
            data_dir=self.data_dir / "test",
            image_transform=self.image_transform,
            mask_transform=self.mask_transform,
            few_shot_list=self.few_shot_list,
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
        data_dir: str | Path,
        image_transform=None,
        mask_transform=None,
        few_shot_list=None,
        embedding_paths_dict=None,
    ):
        self.data_dir = Path(data_dir)
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.embedding_paths_dict = embedding_paths_dict

        self.data = []
        for data_path in sorted(self.data_dir.glob("*.npz")):
            file_stem = data_path.name.split("_")[0]
            if few_shot_list is not None and file_stem not in few_shot_list:
                continue

            if self.embedding_paths_dict:
                self.data.append((data_path, embedding_paths_dict[data_path.stem]))
            else:
                self.data.append((data_path,))

    def add_embedding_paths_dict(self, embedding_paths_dict: dict[str, str]):
        if self.embedding_paths_dict is not None:
            return
        self.embedding_paths_dict = embedding_paths_dict
        self.data = [
            (data[0], embedding_paths_dict[data[0].stem]) for data in self.data
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        image_mask = np.load(data[0])

        image = image_mask["image"]
        mask = image_mask["mask"]
        # 1 indicates presence of the mask, 0 indicates absence
        label = (mask.sum() > 0).astype(np.uint8)

        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        if self.dict_embeddings:
            embs = torch.load(data[1])
            return {"image": image, "mask": mask, "label": label, "embs": embs}
        else:
            return {"image": image, "mask": mask, "label": label, "image_path": data[0]}

if __name__ == "__main__":
    import argparse

    import yaml
    parser = argparse.ArgumentParser(description="Script using YAML config.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration yaml file"
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    data_module = AcouslicAIDataModule(
        task=config["task"],
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        image_transform=None,
        mask_transform=None,
    )
    data_module.prepare_data()
    data_module.setup()