import torch
import torch.nn as nn
from . import modules

class UNETR(nn.Module):
    def __init__(self,
    # encoder params
    in_channels,
    img_size,
    patch_size,
                                    
    # segmentation decoder params
    out_channels):
        super().__init__()
        self.encoder = modules.ViTB16(
            img_size=img_size,
            patch_size=patch_size,
            channels=in_channels
            )
        self.segmentation_decoder = modules.UNETR_decoder(
            ViTB16=self.encoder,
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=list(img_size)
            )
    def forward(self, X):
        segmentation_output = self.segmentation_decoder(X)
        return segmentation_output