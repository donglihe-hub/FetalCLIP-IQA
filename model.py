import logging

import torch
import torch.nn as nn
import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils
from torchmetrics import (
    MetricCollection,
    Accuracy,
    Recall,
    F1Score,
    Precision,
    Specificity,
    ConfusionMatrix,
)

from embeddings import EncoderWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassificationModel(L.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int,
        num_classes: int,
        freeze_encoder: bool,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])

        self.num_classes = num_classes

        # network architecture
        self.encoder = encoder
        self.classifier = nn.Linear(input_dim, num_classes)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # loss function
        self.loss_fn = nn.BCEWithLogitsLoss()

        # optimizer
        self.lr = 3e-4

        # metrics
        self.val_metrics = MetricCollection(
            {
                "accuracy": Accuracy(task="binary"),
                "f1": F1Score(task="binary"),
                "recall": Recall(task="binary"),
                "precision": Precision(task="binary"),
                "specificity": Specificity(task="binary"),
            },
            prefix="val/",
        )
        self.test_metrics = self.val_metrics.clone(prefix="test/")
        self.test_metrics.add_metrics({"confmat": ConfusionMatrix(task="binary")})

        # Embedding visualization [1/3]: save test embeddings for visualization
        # self.test_embeds = []
        # self.test_emb_labels = []

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]

        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]

        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.log("val/loss", loss, prog_bar=True)

        probs = torch.sigmoid(logits)
        self.val_metrics.update(probs, y)

    def on_validation_epoch_end(self):
        results = self.val_metrics.compute()
        self.val_metrics.reset()

        self.log("val/f1", results.pop("val/f1"), prog_bar=True)
        self.log_dict(results)

    def test_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]

        logits = self(x)

        probs = torch.sigmoid(logits)
        self.test_metrics.update(probs, y)

        # Embedding visualization [2/3]: save test embeddings for visualization
        # embeddings = self.encoder(x)
        # embeddings = embeddings.detach().cpu()
        # self.test_embeds.append(embeddings)
        # self.test_emb_labels.append(y)

    def on_test_epoch_end(self):
        results = self.test_metrics.compute()
        self.test_metrics.reset()

        confmat = results.pop("test/confmat").cpu().numpy()
        self.log_dict(results)
        for i in range(2):
            for j in range(2):
                self.log(f"test/confmat_{i}_{j}", confmat[i, j])

        fig, ax = plt.subplots()
        sns.heatmap(confmat, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        self.logger.experiment.log(
            {
                "test/confmat": wandb.Image(fig),
            }
        )
        plt.close(fig)

        # Embedding visualization [3/3]: save test embeddings for visualization
        # test_embs = torch.cat(self.test_embeds, dim=0)
        # test_emb_labels = torch.cat(self.test_emb_labels, dim=0)
        # torch.save(test_embs, "test_embs.pt")
        # torch.save(test_emb_labels, "test_emb_labels.pt")
        # self.test_embs.clear()
        # self.test_emb_labels.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


class SegmentationModel(L.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int,
        num_classes: int,
        freeze_encoder: bool,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])

        self.num_classes = num_classes

        # network architecture
        self.encoder = EncoderWrapper(encoder)
        self.decoder = UNETR(input_dim, num_classes, input_dim=3, init_filters=32)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # loss function
        self.loss_fn = smp.losses.DiceLoss(mode="multilabel", from_logits=True)

        # optimizer
        self.lr = 3e-4

        # validation step outputs
        self.val_test_step_outputs = []

        # metric
        self.test_metrics = MetricCollection(
            {
                "accuracy": Accuracy(task="binary"),
                "f1": F1Score(task="binary"),
                "recall": Recall(task="binary"),
                "precision": Precision(task="binary"),
                "specificity": Specificity(task="binary"),
                "confmat": ConfusionMatrix(task="binary"),
            },
            prefix="test/",
        )

        # threshold
        self.threshold = 224 * 224 * 0.01  # = 501

    def forward(self, x):
        embs = self.encoder(x)
        x = self.decoder([x, *embs.values()])
        return x

    def training_step(self, batch, batch_nb):
        x = batch["image"]
        y = batch["mask"]

        logits = self(x)
        loss = self.loss_fn(logits, y)

        dice = smp_utils.metrics.Fscore(activation="sigmoid")(logits, y)

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/dice", dice, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb):
        x = batch["image"]
        y = batch["mask"]

        logits = self(x)

        self.val_test_step_outputs.append((logits, y))

        return logits, y

    def on_validation_epoch_end(self):
        preds = []
        targets = []
        for outs in self.val_test_step_outputs:
            preds.append(outs[0])
            targets.append(outs[1])

        preds = torch.cat(preds).cpu()
        targets = torch.cat(targets).cpu()

        loss = self.loss_fn(preds, targets)
        dice = smp_utils.metrics.Fscore(activation="sigmoid")(preds, targets)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/dice", dice, prog_bar=True)

        self.val_test_step_outputs.clear()

    def test_step(self, batch, batch_nb):
        x = batch["image"]
        y = batch["mask"]

        logits = self(x)

        self.val_test_step_outputs.append((logits, y))

        probs = torch.sigmoid(logits)
        print(probs.shape, y.shape)
        print("max, min values of probs and y:")
        print(probs.max(), probs.min(), y.max(), y.min())
        cls_preds = ((probs > 0.5).sum((2, 3)) >= self.threshold).int()
        cls_targets = (y.sum((2, 3)) > 0).int()

        self.test_metrics.update(cls_preds, cls_targets)

    def on_test_epoch_end(self):
        results = self.test_metrics.compute()
        self.test_metrics.reset()

        confmat = results.pop("test/confmat").cpu().numpy()
        self.log_dict(results)
        for i in range(2):
            for j in range(2):
                self.log(f"test/confmat_{i}_{j}", confmat[i, j])

        fig, ax = plt.subplots()
        sns.heatmap(confmat, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        self.logger.experiment.log(
            {
                "test/confmat": wandb.Image(fig),
            }
        )
        plt.close(fig)

        preds = []
        targets = []
        for outs in self.val_test_step_outputs:
            preds.append(outs[0])
            targets.append(outs[1])

        preds = torch.cat(preds).cpu()
        targets = torch.cat(targets).cpu()

        test_dice = smp_utils.metrics.Fscore(activation="sigmoid")(preds, targets)
        self.log("test/dice", test_dice)

        self.val_test_step_outputs.clear()

        ## save predicted masks and ground truth masks for visualization
        # from pathlib import Path

        # import torchvision.transforms.functional as TF

        # example_out_folder = Path("outputs/segmentation_examples")
        # example_out_folder.mkdir(exists_ok=True)

        # empty_counts = 0
        # for i in range(preds.size(0)):
        #     pred_mask = (torch.sigmoid(preds[i]) > 0.5).float()
        #     gt_mask = targets[i]
        #     if gt_mask.max() == 1:
        #         pred_save_path = example_out_folder / f"{i}_pred.png"
        #         gt_save_path = example_out_folder / f"{i}_gt.png"
        #         TF.to_pil_image(pred_mask).save(pred_save_path)
        #         TF.to_pil_image(gt_mask).save(gt_save_path)
        #     elif empty_counts < 50:
        #             pred_save_path = example_out_folder / f"empty_{i}_pred.png"
        #             gt_save_path = example_out_folder / f"empty_{i}_gt.png"
        #             TF.to_pil_image(pred_mask).save(pred_save_path)
        #             TF.to_pil_image(gt_mask).save(gt_save_path)
        #         empty_counts += 1

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


"""
REFERENCES:
- https://github.com/tamasino52/UNETR/blob/main/unetr.py#L171
"""


class SingleDeconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, groups=1):
        super().__init__()
        self.block = nn.ConvTranspose2d(
            in_planes,
            out_planes,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
            groups=groups,
        )

    def forward(self, x):
        return self.block(x)


class SingleConv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1):
        super().__init__()
        self.block = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=1,
            padding=((kernel_size - 1) // 2),
            groups=groups,
        )

    def forward(self, x):
        return self.block(x)


class Conv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv2DBlock(in_planes, in_planes, kernel_size, groups=in_planes),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(True),
            SingleConv2DBlock(in_planes, out_planes, 1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)


class Deconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv2DBlock(in_planes, in_planes, groups=in_planes),
            SingleConv2DBlock(in_planes, in_planes, kernel_size, groups=in_planes),
            nn.BatchNorm2d(in_planes),
            nn.ReLU(True),
            SingleConv2DBlock(in_planes, out_planes, 1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)


class SingleDWConv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv2DBlock(in_planes, in_planes, groups=in_planes),
            SingleConv2DBlock(in_planes, out_planes, 1),
        )

    def forward(self, x):
        return self.block(x)


class UNETR(nn.Module):
    def __init__(self, transformer_width, output_dim, input_dim, init_filters):
        super().__init__()

        self.decoder0 = nn.Sequential(
            Conv2DBlock(input_dim, init_filters, 3),
            Conv2DBlock(init_filters, init_filters, 3),
        )

        self.decoder3 = nn.Sequential(
            Deconv2DBlock(transformer_width, 8 * init_filters),
            Deconv2DBlock(8 * init_filters, 4 * init_filters),
            Deconv2DBlock(4 * init_filters, 2 * init_filters),
        )

        self.decoder6 = nn.Sequential(
            Deconv2DBlock(transformer_width, 8 * init_filters),
            Deconv2DBlock(8 * init_filters, 4 * init_filters),
        )

        self.decoder9 = Deconv2DBlock(transformer_width, 8 * init_filters)

        self.decoder12_upsampler = SingleDWConv2DBlock(
            transformer_width, 8 * init_filters
        )

        self.decoder9_upsampler = nn.Sequential(
            Conv2DBlock(16 * init_filters, 8 * init_filters),
            Conv2DBlock(8 * init_filters, 8 * init_filters),
            Conv2DBlock(8 * init_filters, 8 * init_filters),
            SingleDWConv2DBlock(8 * init_filters, 4 * init_filters),
        )

        self.decoder6_upsampler = nn.Sequential(
            Conv2DBlock(8 * init_filters, 4 * init_filters),
            Conv2DBlock(4 * init_filters, 4 * init_filters),
            SingleDWConv2DBlock(4 * init_filters, 2 * init_filters),
        )

        self.decoder3_upsampler = nn.Sequential(
            Conv2DBlock(4 * init_filters, 2 * init_filters),
            Conv2DBlock(2 * init_filters, 2 * init_filters),
            SingleDWConv2DBlock(2 * init_filters, init_filters),
        )

        self.decoder0_header = nn.Sequential(
            Conv2DBlock(2 * init_filters, init_filters),
            Conv2DBlock(init_filters, init_filters),
            SingleConv2DBlock(init_filters, output_dim, 1),
        )

    def forward(self, x):
        z0, z3, z6, z9, z12 = x

        # print(z0.shape, z3.shape, z6.shape, z9.shape, z12.shape)
        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(z0)
        # print(z0.shape, z3.shape, z6.shape, z9.shape, z12.shape)
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))

        return output
