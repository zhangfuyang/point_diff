from diffusers import UNet2DConditionModel, UNet2DModel
import torch.nn as nn
import torch
from transformer import PointDiffusionTransformer

class myModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = PointDiffusionTransformer(
            heads=8, init_scale=0.25, input_channels=3, 
            layers=12, n_ctx=512, output_channels=3, 
            time_token_cond=True, width=512)

    
    def forward(self, x, timestep):
        """
        :param x: an [BS x 2 x N] tensor.
        :param t: an [BS] tensor.
        """
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        if timestep.shape[0] == 1:
            timestep = timestep.repeat(x.shape[0])
        timestep = timestep.to(x.device)
        noise_pred = self.model(x, timestep)
        return noise_pred
    
