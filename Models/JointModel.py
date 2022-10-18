import torch
import torch.nn as nn
from . import modules

class JointModel(nn.Module):
    def __init__(self,
    # encoder params
    in_channels,
    img_size,
    patch_size,
    
    # reconstruction params
    decoder_dim,
    masking_ratio,
    
    # segmentation decoder params
    out_channels):
        super().__init__()
        self.encoder = modules.ViTB16(
            img_size=img_size,
            patch_size=patch_size
        )
        self.segmentation_decoder = modules.UNETR_decoder(
            ViTB16=self.encoder,
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=list(img_size),
            patch_size=patch_size
        )
        self.reconstruction_decoder = modules.MAE_decoder(
            encoder=self.encoder,
            decoder_dim=decoder_dim,
            masking_ratio=masking_ratio
        )

    def forward(self, X):
        recon_loss, reconstruction_output = self.reconstruction_decoder(X)
        segmentation_output = self.segmentation_decoder(X)

        return segmentation_output, recon_loss, reconstruction_output