import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class SpatialAAE(nn.Module):
    def __init__(self, in_channels, latent_channels, num_views=7):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels, 3, padding=1), # Depthwise
            nn.SiLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, in_channels, 1),
            nn.SiLU() 
        )

    def forward(self, x):

        H = self.encoder(x) 
        Rec = self.decoder(H)
        
        return H, Rec

class ParallelAAE(nn.Module):
    def __init__(self, ch_list, compression_ratio=0.5, num_views=7):
        super().__init__()

        self.ch_latent = [int(c * compression_ratio) for c in ch_list]
        
        # P3, P4, P5
        self.aae_p3 = SpatialAAE(ch_list[0], self.ch_latent[0], num_views)
        self.aae_p4 = SpatialAAE(ch_list[1], self.ch_latent[1], num_views)
        self.aae_p5 = SpatialAAE(ch_list[2], self.ch_latent[2], num_views)

    def forward(self, x_list):
        h3, rec3 = self.aae_p3(x_list[0])
        h4, rec4 = self.aae_p4(x_list[1])
        h5, rec5 = self.aae_p5(x_list[2])
        
        return [h3, h4, h5], [rec3, rec4, rec5]