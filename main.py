import argparse
import logging
import json
from pathlib import Path

import yaml
import pandas as pd
import lightning as L
import torchvision.transforms as T
import albumentations as A
import open_clip
from lightning.pytorch.callbacks import ModelCheckpoint, Timer

# package_path = Path(__file__).resolve().parent / "FetalCLIP"
# sys.path.append(package_path)

from data import AcouslicAIDataModule
from model import EncoderWrapper, ClassificationModel, SegmentationModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLASS_NAME_TO_IDX= {
    'Abscent': 0,
    'Present': 1,
}

def get_model_and_transforms(config: dict):
    # Load model configuration
    fetalclip_config_path = config["fetalclip_config_path"]
    fetalclip_weights_path = config["paths"]["fetalclip_weights_path"]
    with open(fetalclip_config_path, "r") as file:
        fetalclip_config = json.load(file)
    open_clip.factory._MODEL_CONFIGS[config["model_name"]] = fetalclip_config

    model, _, image_transform = open_clip.create_model_and_transforms(
        ["model_name"], pretrained=fetalclip_weights_path
    )
    model.visual.eval()

    model = EncoderWrapper(model.visual)

    mask_transform = None
    if config["task"] == "segmentation":
        image_size = fetalclip_config["vision_cfg"]["image_size"]
        mask_transform = T.Compose(
            [
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
                T.ToTensor(),
            ]
        )

    return model, image_transform, mask_transform


def main(config):
    model_name = config["model_name"]
    task = config["task"]

    model_name, encoder, image_transform, mask_transform = get_model_and_transforms(
        config
    )

    encoder.eval()
    encoder = encoder.cuda()

    exp_logs_dir = Path(config["experiment_logs_dir"]) / task
    exp_logs_path = exp_logs_dir / f"{model_name}.csv"

    data_module = AcouslicAIDataModule(
        task=task,
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        image_transform=image_transform,
        mask_transform=mask_transform,
    )
    data_module.prepare_data()
    data_module.setup()
    data_module.prepare_embeddings(encoder)
    
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    test_dataloader = data_module.test_dataloader()

    num_trials = config["num_trials"]
    for trial in range(num_trials):
        if exp_logs_path.exists():
            df_logs = pd.read_csv(exp_logs_path)
            if trial in df_logs["trial"].values:
                continue
        
        logger.info("*" * 30)

        num_classes = 1 if len(CLASS_NAME_TO_IDX) == 2 else len(CLASS_NAME_TO_IDX)
        if task == "classification":
            model = ClassificationModel(input_dim, num_classes)
        elif task == "segmentation":
            model = SegmentationModel(
                encoder.transformer.width, num_classes, 3
            )

        checkpoint = ModelCheckpoint(
            monitor="val_loss", mode="min", dirpath=Path("fetalclip" / task / trial),
        )

        trainer = L.Trainer(
            devices=1,
            accelerator="gpu",
            max_epochs=config["max_epochs"],
            callbacks=[checkpoint, Timer()],
        )

        trainer.fit(model, train_dataloader, val_dataloader)
        trainer.test(model, dataloaders=test_dataloader, ckpt_path="best")

        model.test_metrics = {
            "trial": trial,
            **{key: val.item() for key, val in model.test_metrics.items()},
        }

        # Save results to CSV
        df_new = pd.DataFrame([model.test_metrics])
        if exp_logs_path.exists():
            df_new.to_csv(exp_logs_path, mode="a", header=False, index=False)
        else:
            exp_logs_path.parent.mkdir(parents=True)
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

    main(config)
