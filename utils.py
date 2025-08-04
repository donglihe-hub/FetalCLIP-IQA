import logging
from typing import Optional
from pathlib import Path

from jsonargparse.typing import NonNegativeInt
from rich.logging import RichHandler


class Settings:
    def __init__(
        self,
        model_name: str,
        task: str,
        data_dir: str,
        output_dir: str,
        fetalclip_config_path: Optional[str] = None,
        fetalclip_weights_path: Optional[str] = None,
        num_workers: NonNegativeInt = 8,
        batch_size: NonNegativeInt = 32,
        max_epochs: NonNegativeInt = 5,
        num_trials: NonNegativeInt = 5,
        use_augmentation: bool = True,
        freeze_encoder: bool = True,
        use_lora: bool = True,
        debug_mode: bool = False,
    ):
        """
        Settings for FetalCLIP model training and evaluation.

        Args:
            model_name: Name of the model to use (e.g., "fetalclip")
            task: Type of task (e.g., "classification")
            data_dir: Directory containing the dataset
            output_dir: Directory to save outputs
            fetalclip_config_path: Path to FetalCLIP config file
            fetalclip_weights_path: Path to FetalCLIP weights file
            num_workers: Number of workers for data loading
            batch_size: Batch size for training
            max_epochs: Maximum number of training epochs
            num_trials: Number of trials/runs
            store_embeddings: Whether to store embeddings
            use_augmentation: Whether to use data augmentation
            freeze_encoder: Whether to freeze the encoder weights
            use_lora: Whether to use LoRA (Low-Rank Adaptation)
        """
        # Required parameters
        assert task in ["classification", "segmentation"], "Invalid task type"

        # Optional but recommended parameters
        if model_name.lower() == "fetalclip":
            assert (
                fetalclip_config_path is not None
            ), "fetalclip_config_path must be specified for FetalCLIP"
            assert (
                fetalclip_weights_path is not None
            ), "fetalclip_weights_path must be specified for FetalCLIP"

        self.model_name = model_name
        self.task = task
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.fetalclip_config_path = Path(fetalclip_config_path)
        self.fetalclip_weights_path = Path(fetalclip_weights_path)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.num_trials = num_trials
        self.use_augmentation = use_augmentation
        self.freeze_encoder = freeze_encoder
        self.use_lora = use_lora
        self.debug_mode = debug_mode

    def __getitem__(self, key):
        return getattr(self, key)

    def _to_dict(self) -> dict[str]:
        """Convert settings to dict"""
        result = {}
        for key, value in vars(self).items():
            if isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result


def init_logger(verbose: bool = False):
    logger = logging.getLogger()
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)

    logger.handlers.clear()
    handler = RichHandler()
    handler.setLevel(level)
    logger.addHandler(handler)
