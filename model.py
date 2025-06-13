import os
import json

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torchvision.transforms as T
import albumentations as A
import open_clip
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as smp_utils

from PIL import Image

with open('config.json', 'r') as file:
    config = json.load(file)

with open(PATH_TRAIN_VAL_SPLIT, 'r') as file:
    DICT_LIST_PID = json.load(file)

with open('test_id.json', 'r') as file:
    LIST_TEST_PID = json.load(file)
LIST_TEST_PID = list(LIST_TEST_PID.keys())

LIST_CLASSES = ['head']
NUM_CLASSES  = len(LIST_CLASSES)


class ClassificationModel(L.LightningModule):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.head = torch.nn.Linear(input_dim, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.validation_step_outputs = []
    
    def forward(self, x):
        x = self.head(x)
        return x

    def training_step(self, batch, batch_nb):
        _, y, embs = batch

        pred = self.forward(embs)
        
        loss = self.criterion(pred, y)
        pred = torch.argmax(pred, 1)
        acc = Accuracy(task="multiclass", num_classes=self.num_classes, top_k=1, average='macro')(pred.to('cpu'), y.to('cpu')).item()
        f1  = F1Score(task="multiclass", num_classes=self.num_classes, top_k=1, average='macro')(pred.to('cpu'), y.to('cpu')).item()

        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_f1', f1, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_nb):
        _, y, embs = batch

        pred = self.forward(embs)
        
        self.validation_step_outputs.append((pred,y))
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

        loss = self.criterion(preds, targets)

        acc = Accuracy(task="multiclass", num_classes=self.num_classes, top_k=1, average='macro')(preds,targets)
        f1 = F1Score(task="multiclass", num_classes=self.num_classes, top_k=1, average='macro')(preds,targets)

        acc_top2 = Accuracy(task="multiclass", num_classes=self.num_classes, top_k=2, average='macro')(preds,targets)
        acc_top3 = Accuracy(task="multiclass", num_classes=self.num_classes, top_k=3, average='macro')(preds,targets)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc_top2', acc_top2, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc_top3', acc_top3, on_step=False, on_epoch=True, prog_bar=True)
        
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        preds = []
        targets = []

        for outs in self.validation_step_outputs:
            preds.append(outs[0])
            targets.append(outs[1])
        
        preds = torch.cat(preds).cpu()
        targets = torch.cat(targets).cpu()

        loss = self.criterion(preds, targets)

        macc = Accuracy(task="multiclass", num_classes=self.num_classes, top_k=1, average='macro')(preds,targets)
        mf1 = F1Score(task="multiclass", num_classes=self.num_classes, top_k=1, average='macro')(preds,targets)

        macc_top2 = Accuracy(task="multiclass", num_classes=self.num_classes, top_k=2, average='macro')(preds,targets)

        self.test_metrics = {
            'test_loss': loss,
            'test_f1': mf1,
            'test_acc': macc,
            'test_acc_top2': macc_top2,
        }

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_f1', mf1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_acc', macc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_acc_top2', macc_top2, on_step=False, on_epoch=True, prog_bar=True)

        acc = Accuracy(task="multiclass", num_classes=self.num_classes, top_k=1, average='none')(preds,targets)
        f1 = F1Score(task="multiclass", num_classes=self.num_classes, top_k=1, average='none')(preds,targets)

        # long_string = f"{mf1.item()},{macc.item()},{macc_top2.item()},{','.join([str(x.item()) for x in f1])},{','.join([str(x.item()) for x in acc])},{','.join([str(x.item()) for x in acc_top2])}"
        
        for key, val in DICT_CLSNAME_TO_CLSINDEX.items():
            self.test_metrics[f'test_acc_{key}'] = acc[val]
            self.test_metrics[f'test_f1_{key}']  =  f1[val]

        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.head.parameters(), lr=3e-4, weight_decay=0.01)

class SegmentationModel(L.LightningModule):
    def __init__(self, transformer_width, num_classes, input_dim, init_filters=32):
        super().__init__()
        self.num_classes = num_classes
        self.model = UNETR(transformer_width, num_classes, input_dim, init_filters)
        self.criterion = smp.losses.DiceLoss(mode='multilabel', from_logits=True)
        self.validation_step_outputs = []
    
    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y, embs = batch

        pred = self.forward([x, *embs])
        
        loss = self.criterion(pred, y)
        
        dsc = smp_utils.metrics.Fscore(activation='sigmoid')(pred, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_dsc', dsc, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_nb):
        x, y, embs = batch
        
        pred = self.forward([x, *embs])
        
        self.validation_step_outputs.append((pred,y))
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

        loss = self.criterion(preds, targets)
        
        dsc = smp_utils.metrics.Fscore(activation='sigmoid')(preds, targets)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_dsc', dsc, on_step=False, on_epoch=True, prog_bar=True)
        
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        preds = []
        targets = []

        for outs in self.validation_step_outputs:
            preds.append(outs[0])
            targets.append(outs[1])
        
        preds = torch.cat(preds).cpu()
        targets = torch.cat(targets).cpu()

        test_loss = self.criterion(preds, targets)

        test_dsc = smp_utils.metrics.Fscore(activation='sigmoid')(preds, targets)
        self.log('test_loss', test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_dsc', test_dsc, on_step=False, on_epoch=True, prog_bar=True)

        self.test_metrics = {'test_loss': test_loss, 'test_dsc': test_dsc}

        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        trainable_params = (
            param for name, param in self.named_parameters()
            if not name.startswith('model.transformer')
        )
        for name, param in self.named_parameters():
            if name.startswith('model.transformer'):
                param.requires_grad = False
        return torch.optim.AdamW(trainable_params, lr=3e-4, weight_decay=0.01)

'''
REFERENCES:
- https://github.com/tamasino52/UNETR/blob/main/unetr.py#L171
'''

class SingleDeconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, groups=1):
        super().__init__()
        self.block = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0, groups=groups)

    def forward(self, x):
        return self.block(x)


class SingleConv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1):
        super().__init__()
        self.block = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size - 1) // 2), groups=groups)

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

        self.decoder0 = \
            nn.Sequential(
                Conv2DBlock(input_dim, init_filters, 3),
                Conv2DBlock(init_filters, init_filters, 3)
            )

        self.decoder3 = \
            nn.Sequential(
                Deconv2DBlock(transformer_width, 8*init_filters),
                Deconv2DBlock(8*init_filters, 4*init_filters),
                Deconv2DBlock(4*init_filters, 2*init_filters)
            )

        self.decoder6 = \
            nn.Sequential(
                Deconv2DBlock(transformer_width, 8*init_filters),
                Deconv2DBlock(8*init_filters, 4*init_filters),
            )

        self.decoder9 = \
            Deconv2DBlock(transformer_width, 8*init_filters)

        self.decoder12_upsampler = \
            SingleDWConv2DBlock(transformer_width, 8*init_filters)

        self.decoder9_upsampler = \
            nn.Sequential(
                Conv2DBlock(16*init_filters, 8*init_filters),
                Conv2DBlock(8*init_filters, 8*init_filters),
                Conv2DBlock(8*init_filters, 8*init_filters),
                SingleDWConv2DBlock(8*init_filters, 4*init_filters)
            )

        self.decoder6_upsampler = \
            nn.Sequential(
                Conv2DBlock(8*init_filters, 4*init_filters),
                Conv2DBlock(4*init_filters, 4*init_filters),
                SingleDWConv2DBlock(4*init_filters, 2*init_filters)
            )

        self.decoder3_upsampler = \
            nn.Sequential(
                Conv2DBlock(4*init_filters, 2*init_filters),
                Conv2DBlock(2*init_filters, 2*init_filters),
                SingleDWConv2DBlock(2*init_filters, init_filters)
            )

        self.decoder0_header = \
            nn.Sequential(
                Conv2DBlock(2*init_filters, init_filters),
                Conv2DBlock(init_filters, init_filters),
                SingleConv2DBlock(init_filters, output_dim, 1)
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