from pathlib import Path
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from torch.utils.data import DataLoader
from tqdm import tqdm

class HF_VisualEncoderWithHooks(nn.Module):
    """
    REFERENCES:
    - https://github.com/huggingface/pytorch-image-models/blob/2703d155c88d27bba9a1f465f5489a7947ffc313/timm/models/vision_transformer.py#L414
    """
    def __init__(self, visual_encoder):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.hooks = []
        self.intermediate_outputs = {}

        self.width = self.visual_encoder.transformer.width
        self.grid_size = self.visual_encoder.grid_size
        
        # Register hooks when the class is initialized
        self._register_hooks()
    
    def _register_hooks(self):
        """
        Register hooks for the layers specified in self.layers_to_hook.
        """
        n_blocks = len(self.visual_encoder.transformer.resblocks)
        for layer_idx in [n_blocks // i - 1 for i in range(1,5)]:
            layer = self.visual_encoder.transformer.resblocks[layer_idx]
            hook = layer.register_forward_hook(self._get_intermediate_output(f'layer_{layer_idx+1}'))
            self.hooks.append(hook)

    def _get_intermediate_output(self, layer_name):
        """
        Hook function to capture the intermediate output.
        """
        def hook(module, input, output):
            self.intermediate_outputs[layer_name] = output
        return hook

    def forward(self, x):
        """
        Perform the forward pass while capturing intermediate outputs.
        """
        # Reset intermediate outputs before forward pass
        self.intermediate_outputs = {}
        
        # Perform the forward pass of the VisionTransformer
        output = self.visual_encoder(x)

        list_keys = sorted(list(self.intermediate_outputs.keys()), key=lambda x: int(x.split('_')[1]))
        intermediate_outputs = [
            self.intermediate_outputs[key].permute(1,0,2)[1:]
            for key in list_keys
        ]
        
        return output, intermediate_outputs

    def remove_hooks(self):
        """
        Remove all hooks after usage.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class TIMM_VisualEncoderWithHooks(nn.Module):
    """
    REFERENCES:
    - https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/transformer.py#L434
    """
    def __init__(self, visual_encoder):
        super(TIMM_VisualEncoderWithHooks, self).__init__()
        self.visual_encoder = visual_encoder
        self.hooks = []
        self.intermediate_outputs = {}

        self.width = self.visual_encoder.trunk.embed_dim
        self.grid_size = self.visual_encoder.trunk.patch_embed.grid_size
        
        # Register hooks when the class is initialized
        self._register_hooks()
    
    def _register_hooks(self):
        """
        Register hooks for the layers specified in self.layers_to_hook.
        """
        n_blocks = len(self.visual_encoder.trunk.blocks)
        for layer_idx in [n_blocks // i - 1 for i in range(1,5)]:
            layer = self.visual_encoder.trunk.blocks[layer_idx]
            hook = layer.register_forward_hook(self._get_intermediate_output(f'layer_{layer_idx+1}'))
            self.hooks.append(hook)

    def _get_intermediate_output(self, layer_name):
        """
        Hook function to capture the intermediate output.
        """
        def hook(module, input, output):
            self.intermediate_outputs[layer_name] = output
        return hook

    def forward(self, x):
        """
        Perform the forward pass while capturing intermediate outputs.
        """
        # Reset intermediate outputs before forward pass
        self.intermediate_outputs = {}
        
        # Perform the forward pass of the VisionTransformer
        output = self.visual_encoder(x)

        list_keys = sorted(list(self.intermediate_outputs.keys()), key=lambda x: int(x.split('_')[1]))
        intermediate_outputs = [
            self.intermediate_outputs[key].permute(1,0,2)[1:]
            for key in list_keys
        ]
        
        return output, intermediate_outputs

    def remove_hooks(self):
        """
        Remove all hooks after usage.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

class EncoderWrapper(nn.Module):
    def __init__(self, visual_encoder):
        super().__init__()

        if isinstance(visual_encoder, open_clip.transformer.VisionTransformer):
            self.transformer = HF_VisualEncoderWithHooks(visual_encoder)
        elif isinstance(visual_encoder, open_clip.timm_model.TimmModel):
            self.transformer = TIMM_VisualEncoderWithHooks(visual_encoder)
    
    def forward(self, x):
        with torch.no_grad():
            z = self.transformer(x)
        
            z3, z6, z9, z12 = z[1]

            z3 = z3.permute(1,2,0).view(-1, self.transformer.width, *self.transformer.grid_size)
            z6 = z6.permute(1,2,0).view(-1, self.transformer.width, *self.transformer.grid_size)
            z9 = z9.permute(1,2,0).view(-1, self.transformer.width, *self.transformer.grid_size)
            z12 = z12.permute(1,2,0).view(-1, self.transformer.width, *self.transformer.grid_size)
            
            z3 = F.interpolate(z3, size=(14, 14), mode='bilinear', align_corners=False)
            z6 = F.interpolate(z6, size=(14, 14), mode='bilinear', align_corners=False)
            z9 = F.interpolate(z9, size=(14, 14), mode='bilinear', align_corners=False)
            z12 = F.interpolate(z12, size=(14, 14), mode='bilinear', align_corners=False)
        
        return {
            'z3': z3, 'z6': z6,
            'z9': z9, 'z12': z12,
        }

def generate_embeddings(model: L.LightningModule, dataloader: DataLoader, task: str):
    if len(dataloader.dataset) == 0:
        return
    
    for items in tqdm(dataloader, desc='Extracting embeddings'):
        images = items["image"].to('cuda')

        with torch.no_grad():
            z_batch = model(images)

        for i, emb_path in enumerate(items["emb_path"]):
            if task == "classification":
                emb = z_batch[i].detach().cpu()
            elif task == "segmentation":
                emb = {k: z_batch[k][i].detach().cpu() for k in ["z3", "z6", "z9", "z12"]}
            torch.save(emb, emb_path)
