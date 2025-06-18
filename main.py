import os
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

import yaml
import pandas as pd
import lightning as L
import timm
import torch
import torchvision.transforms as T
import open_clip
from lightning.pytorch.callbacks import ModelCheckpoint, Timer, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from data import AcouslicAIDataModule
from model import ClassificationModel, SegmentationModel

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set to the GPU you want to use

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLASS_NAME_TO_IDX= {
    'Abscent': 0,
    'Present': 1,
}

def get_model_and_transforms(model_name: str, config: dict):
    if model_name == "fetalclip":
        # Load model configuration
        fetalclip_config_path = config["fetalclip_config_path"]
        fetalclip_weights_path = config["fetalclip_weights_path"]
        with open(fetalclip_config_path, "r") as file:
            fetalclip_config = json.load(file)
        open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = fetalclip_config

        model, _, image_transform = open_clip.create_model_and_transforms(
            "FetalCLIP", pretrained=fetalclip_weights_path
        )
        model = model.visual

        image_size = fetalclip_config["vision_cfg"]["image_size"]
    elif model_name == "resnet":
        model_name = "resnet50"
    elif model_name == "densenet":
        model_name = "densenet121"
    elif model_name == "mobilenet":
        model_name = "mobilenetv3_small_100"
    elif model_name == "efficientnet":
        model_name = "efficientnet_b0"
    elif model_name == "vgg":
        model_name = "vgg16"
    elif model_name == "vit":
        model_name = "vit_large_patch16_224.augreg_in21k_ft_in1k"
    else:
        raise ValueError(f"Unsupported model name: {config['model_name']}")
    
    if model_name != "fetalclip":
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        model_config = resolve_data_config({}, model=model)
        image_transform = create_transform(**model_config)
        image_size = model.default_cfg['input_size'][-1]

    mask_transform = None
    if config["task"] == "segmentation":
        mask_transform = T.Compose(
            [
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
                T.ToTensor(),
            ]
        )

    return model, image_transform, mask_transform


def main(config):
    run_name_prefix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task = config["task"]
    model_name = config["model_name"].lower()
    encoder, image_transform, mask_transform = get_model_and_transforms(
        model_name, config
    )

    if task == "classification":
        if model_name == "fetalclip":
            input_dim = encoder.proj.shape[1]
        elif model_name in ["resnet", "densenet", "efficientnet"]:
            input_dim = encoder.num_features
        elif model_name in ["mobilenet", "vgg"]:
            input_dim = encoder.head_hidden_size
        elif model_name == "vit":
            input_dim = encoder.embed_dim
    elif task == "segmentation":
        if model_name == "fetalclip":
            transformer_width = encoder.transformer.width

    encoder = encoder.cuda()

    exp_logs_path = Path(config["output_dir"]) / "experiment_logs" / task / f"{model_name}_{run_name_prefix}.csv"

    data_module = AcouslicAIDataModule(
        task=task,
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        image_transform=image_transform,
        mask_transform=mask_transform,
        use_augmentation=config["use_augmentation"],
    )
    data_module.prepare_data()

    if config["store_embeddings"] and model_name == "fetalclip":
        data_module.setup("embeddings")
        data_module.prepare_embeddings(encoder)
        encoder = None
    data_module.setup("fit")
    
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    test_dataloader = data_module.test_dataloader()

    num_trials = config["num_trials"]
    for trial in range(num_trials):
        if exp_logs_path.exists():
            df_logs = pd.read_csv(exp_logs_path)
            if trial in df_logs["trial"].values:
                continue
        run_name = f"{model_name}_{task}_{run_name_prefix}_trial_{trial}"
        logger.info(f"Starting trial {trial + 1}/{num_trials} for task {task}")

        # binary classification
        num_classes = 1  # if len(CLASS_NAME_TO_IDX) == 2 else len(CLASS_NAME_TO_IDX)
        if task == "classification":
            model = ClassificationModel(encoder, input_dim, num_classes, freeze_encoder=config["freeze_encoder"])
        elif task == "segmentation":
            model = SegmentationModel(
                encoder, transformer_width, num_classes, 3, freeze_encoder=config["freeze_encoder"])
        torch.set_float32_matmul_precision('high')
        # model = torch.compile(model)

        checkpoint = ModelCheckpoint(
            monitor="val_loss", mode="min", dirpath=Path(config["output_dir"]) / "fetalclip" / task / model_name / run_name_prefix / str(trial),
        )
        earlystop = EarlyStopping(monitor="val_loss", patience=10, mode="min")
        timer = Timer()
        callbacks = [checkpoint, earlystop, timer]

        wandb_logger = WandbLogger(
            project="Finetune-FetalCLIP",
            name=run_name,
            save_dir=Path(config["output_dir"]) / "fetalclip" / task / model_name / run_name_prefix / str(trial),
        )

        trainer = L.Trainer(
            devices=1,
            accelerator="gpu",
            max_epochs=config["max_epochs"],
            logger=wandb_logger,
            callbacks=callbacks,
        )

        trainer.fit(model, train_dataloader, val_dataloader)
        results = trainer.test(dataloaders=test_dataloader, ckpt_path="best")

        wandb_logger.experiment.log({"train_time": timer.time_elapsed("train")})

        results = {"trial": trial, **results[0]}

        # Save results to CSV
        df_new = pd.DataFrame([results])
        if exp_logs_path.exists():
            df_new.to_csv(exp_logs_path, mode="a", header=False, index=False)
        else:
            exp_logs_path.parent.mkdir(parents=True, exist_ok=True)
            df_new.to_csv(exp_logs_path, mode="w", header=True, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script using YAML config.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration yaml file"
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    logger.info(config)
    try:
        main(config)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()