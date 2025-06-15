import logging

import torch
import torch.nn as nn
import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from monai.losses import DiceLoss
from torch_flops import TorchFLOPsByFX
from torchmetrics import (
    MetricCollection,
    Accuracy,
    Recall,
    F1Score,
    Precision,
    Specificity,
    ConfusionMatrix,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassificationModel(L.LightningModule):
    def __init__(self, encoder: None | nn.Module, input_dim: int, num_classes: int, freeze_encoder: bool = True):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])
        self.num_classes = num_classes
        self.encoder = encoder
        self.output = nn.Linear(input_dim, num_classes)

        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder and self.encoder is not None:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr = 3e-4

        self.val_metrics = MetricCollection(
            {
                "accuracy": Accuracy(task="binary"),
                "recall": Recall(task="binary"),
                "f1": F1Score(task="binary"),
                "precision": Precision(task="binary"),
                "specificity": Specificity(task="binary"),
                "confmat": ConfusionMatrix(task="binary"),
            }
        )
        self.test_metrics = self.val_metrics.clone(prefix="test_")

    def forward(self, x):
        if self.encoder is not None:
            x = self.encoder(x)
        x = self.output(x)
        return x

    # def on_fit_start(self):
    #     n_total = sum(p.numel() for p in self.parameters())
    #     n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
    #     logger.info(f"Total: {n_total}, Trainable: {n_trainable}")

    #     if self.encoder is not None:
    #         model = nn.Sequential(
    #             self.encoder,
    #             self.output,
    #         )
    #     else:
    #         model = self.output

    #     if self.encoder is not None:
    #         x = torch.randn(1, 3, 224, 224).cuda()
    #     else:
    #         x = torch.randn(1, 768).cuda()

    #     with torch.no_grad():
    #         for _ in range(20):
    #             model(x)
        
    #     flops_counter = TorchFLOPsByFX(model)

    #     flops_counter.propagate(x)
    #     total_flops = flops_counter.print_total_flops(show=False)
    #     max_memory = flops_counter.print_max_memory(show=False)

    #     time_list = []
    #     for _ in range(50):
    #         flops_counter.propagate(x)
    #         time_list.append(flops_counter.print_total_time(show=False))
    #     average_time = sum(time_list) / len(time_list)

    #     self.logger.experiment.log({"flops": total_flops, "max_memory": max_memory, "average_time": average_time})
    #     logger.info(f"FLOPs: {total_flops}")
    #     logger.info(f"Max Memory: {max_memory}")
    #     logger.info(f"Average Time: {average_time}")


    def training_step(self, batch, batch_idx):
        if self.encoder is None:
            x = batch["embs"]
        else:
            x = batch["image"]
        y = batch["label"]

        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.encoder is None:
            x = batch["embs"]
        else:
            x = batch["image"]
        y = batch["label"]

        logits = self(x)

        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss)

        probs = torch.sigmoid(logits)
        self.val_metrics.update(probs, y)

    def on_validation_epoch_end(self):
        results = self.val_metrics.compute()
        self.val_metrics.reset()

        confmat = results.pop("confmat").cpu().numpy()
        self.log_dict(results, prog_bar=True)
        for i in range(2):
            for j in range(2):
                self.log(f"confusion_matrix_{i}_{j}", confmat[i, j])

        fig, ax = plt.subplots()
        sns.heatmap(confmat, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        self.logger.experiment.log(
            {
                "confusion_matrix": wandb.Image(fig),
            }
        )
        plt.close(fig)

    def test_step(self, batch, batch_idx):
        if self.encoder is None:
            x = batch["embs"]
        else:
            x = batch["image"]
        y = batch["label"]

        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("test_loss", loss, prog_bar=True)

        probs = torch.sigmoid(logits)
        self.test_metrics.update(probs, y)

    def on_test_epoch_end(self):
        results = self.test_metrics.compute()
        self.test_metrics.reset()

        confmat = results.pop("test_confmat").cpu().numpy()
        self.log_dict(results)
        for i in range(2):
            for j in range(2):
                self.log(f"test_confusion_matrix_{i}_{j}", confmat[i, j])

        fig, ax = plt.subplots()
        sns.heatmap(confmat, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        self.logger.experiment.log(
            {
                "confusion_matrix": wandb.Image(fig),
            }
        )
        plt.close(fig)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.output.parameters(), lr=self.lr)


class SegmentationModel(L.LightningModule):
    def __init__(self, encoder, transformer_width, num_classes, input_dim, init_filters=32):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.output = UNETR(transformer_width, num_classes, input_dim, init_filters)

        self.loss_fn = DiceLoss(sigmoid=True)  # DiceCELoss(sigmoid=True)
        self.lr = 3e-4

        self.validation_step_outputs = []

    def forward(self, x):
        x = self.encoder(x)
        x = self.output(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y, embs = batch

        pred = self.forward([x, *embs])

        loss = self.loss_fn(pred, y)

        # dsc = smp_utils.metrics.Fscore(activation='sigmoid')(pred, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log('train_dsc', dsc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb):
        x, y, embs = batch

        pred = self.forward([x, *embs])

        self.validation_step_outputs.append((pred, y))
        return pred, y

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def on_validation_epoch_end(self):
        preds = []
        targets = []

        for outs in self.validation_step_outputs:
            preds.append(outs[0])
            targets.append(outs[1])

        preds = torch.cat(preds).cpu()
        targets = torch.cat(targets).cpu()

        loss = self.loss_fn(preds, targets)

        # dsc = smp_utils.metrics.Fscore(activation='sigmoid')(preds, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('val_dsc', dsc, on_step=False, on_epoch=True, prog_bar=True)

        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        preds = []
        targets = []

        for outs in self.validation_step_outputs:
            preds.append(outs[0])
            targets.append(outs[1])

        preds = torch.cat(preds).cpu()
        targets = torch.cat(targets).cpu()

        test_loss = self.loss_fn(preds, targets)

        # test_dsc = smp_utils.metrics.Fscore(activation='sigmoid')(preds, targets)
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('test_dsc', test_dsc, on_step=False, on_epoch=True, prog_bar=True)

        self.test_metrics = {"test_loss": test_loss, "test_dsc": test_dsc}

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        trainable_params = (
            param
            for name, param in self.named_parameters()
            if not name.startswith("model.transformer")
        )
        for name, param in self.named_parameters():
            if name.startswith("model.transformer"):
                param.requires_grad = False
        return torch.optim.AdamW(trainable_params, lr=self.lr, weight_decay=0.01)


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
