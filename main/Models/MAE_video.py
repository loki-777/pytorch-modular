import torch
import torch.nn as nn
from . import modules

class MAEVideoModel(nn.Module):
    def __init__(self,
    # encoder params
    in_channels,
    img_size,
    patch_size,
    
    # reconstruction params
    decoder_dim,
    masking_ratio,
    ):
        super().__init__()
        self.encoder = modules.ViTB16(
            img_size=img_size,
            patch_size=patch_size,
            channels=in_channels
        )
        self.reconstruction_decoder = modules.MAE_decoder(
            encoder=self.encoder,
            decoder_dim=decoder_dim,
            masking_ratio=masking_ratio
        )

    def forward(self, X):
        recon_loss, reconstruction_output = self.reconstruction_decoder(X)

        return _, recon_loss, reconstruction_output