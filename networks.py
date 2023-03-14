import torch
from torch import nn


class DataProcessingNet(nn.Module):

    def __init__(
            self, global_features_dim: int, img_features_dim: int, map_size: int,
            global_embed_dim=9, output_channels=128
    ):
        super().__init__()
        self.global_embed = nn.Linear(global_features_dim, global_embed_dim)
        self.global_embed_cnn = nn.Conv2d(global_embed_dim, global_embed_dim, kernel_size=1)
        self.output_cnn = nn.Conv2d(global_embed_dim + img_features_dim, output_channels, kernel_size=1)
        self.map_size = map_size

    def forward(self, observations: dict[str, torch.Tensor]):
        global_obs = observations['global']
        img_obs = observations['img']

        glb = self.global_embed(global_obs)
        glb = glb.view(-1, 1, 1).expand(-1, self.map_size, self.map_size)
        glb = self.global_embed_cnn(glb)
        x = torch.concatenate([glb, img_obs], dim=0)
        x = self.output_cnn(x)
        return x
