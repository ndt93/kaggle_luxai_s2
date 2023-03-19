from typing import Callable, Tuple

import gym
import numpy.typing as npt
import torch
from torch import nn
import torchvision as tv
from stable_baselines3.common.distributions import MultiCategoricalDistribution
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


class DataProcessingNet(nn.Module):

    def __init__(
            self, global_features_dim: int, img_features_dim: int, map_size: int,
            global_embed_dim: int, output_channels: int
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
        glb = glb.view(global_obs.shape[0], -1, 1, 1).expand(-1, -1, self.map_size, self.map_size)
        glb = self.global_embed_cnn(glb)
        x = torch.concatenate([glb, img_obs], dim=1)
        x = self.output_cnn(x)
        return x


class ResBlock(nn.Module):

    def __init__(self, map_size, n_channels, squeeze_dim):
        super().__init__()
        self.cnn_net = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=5, padding=2, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(n_channels, n_channels, kernel_size=5, padding=2, stride=1),
            nn.LeakyReLU(),
            tv.ops.SqueezeExcitation(n_channels, squeeze_dim)
        )
        self.out_activation = nn.LeakyReLU()

    def forward(self, inp):
        x = self.cnn_net(inp)
        x = x + inp
        x = self.out_activation(x)
        return x


class ResNet(nn.Module):

    def __init__(self, map_size, n_channels, squeeze_dim, n_blocks):
        super().__init__()
        self.res_blocks = nn.Sequential(
            *[ResBlock(map_size, n_channels, squeeze_dim) for _ in range(n_blocks)]
        )

    def forward(self, inp):
        return self.res_blocks(inp)


class PixelFeaturesExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Dict, features_dim=128,
                 global_embed_dim=9, map_size=48, resnet_squeeze_dim=9, n_resnet_blocks=8):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        global_inp_dim = observation_space['global'].shape[0]
        img_inp_dim = observation_space['img'].shape[0]
        self.data_proc_net = DataProcessingNet(
            global_features_dim=global_inp_dim,
            img_features_dim=img_inp_dim,
            map_size=map_size,
            global_embed_dim=global_embed_dim,
            output_channels=features_dim
        )
        self.res_net = ResNet(
            map_size=map_size,
            n_channels=features_dim,
            squeeze_dim=resnet_squeeze_dim,
            n_blocks=n_resnet_blocks
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.data_proc_net(observations)
        x = self.res_net(x)
        return x


class PixelPolicyActionHead(nn.Module):

    def __init__(self, features_dim, n_actions):
        super().__init__()
        self.cnn_flatten = nn.Sequential(
            nn.Conv2d(features_dim, n_actions, 1),
            nn.Flatten()
        )

    def forward(self, inp):
        return self.cnn_flatten(inp)


class PixelPolicyActionNet(nn.Module):

    def __init__(self, features_dim, n_factory_actions, n_robot_actions):
        super().__init__()
        self.factory_head = PixelPolicyActionHead(features_dim, n_factory_actions)
        self.heavy_head = PixelPolicyActionHead(features_dim, n_robot_actions)
        self.light_head = PixelPolicyActionHead(features_dim, n_robot_actions)

    def forward(self, inp):
        f = self.factory_head(inp)
        h = self.heavy_head(inp)
        l = self.light_head(inp)
        return torch.concatenate([f, h, l], dim=1)


class PixelPolicyValueHead(nn.Module):

    def __init__(self, map_size):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(map_size)

    def forward(self, inp):
        x = self.avg_pool(inp)
        x = torch.flatten(x, start_dim=1)
        return x


class PixelPolicyExtractor(nn.Module):

    def __init__(self, features_dim: int, last_layer_dim_pi: int, last_layer_dim_vf: int,
                 n_factory_actions: int, n_robot_actions: int, map_size: int):
        super().__init__()
        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.policy_net = PixelPolicyActionNet(features_dim, n_factory_actions, n_robot_actions)
        self.value_net = PixelPolicyValueHead(map_size)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


class MultiCategoricalDistribution2d(MultiCategoricalDistribution):

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Identity()


class PixelPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.MultiDiscrete,
            lr_schedule: Callable[[float], float],
            n_factory_actions: int,
            n_robot_actions: int,
            map_size: int,
            features_extractor_class=PixelFeaturesExtractor,
            *args,
            **kwargs,
    ):
        self.n_factory_actions = n_factory_actions
        self.n_robot_actions = n_robot_actions
        self.map_size = map_size
        super().__init__(observation_space,
                         action_space,
                         lr_schedule,
                         features_extractor_class=features_extractor_class,
                         *args, **kwargs)
        self.features_dim = self.features_extractor_kwargs['features_dim']
        self.action_dist = MultiCategoricalDistribution2d(action_space.nvec)
        self._build(lr_schedule)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = PixelPolicyExtractor(
            features_dim=self.features_dim,
            last_layer_dim_pi=1,
            last_layer_dim_vf=self.features_dim,
            n_factory_actions=self.n_factory_actions,
            n_robot_actions=self.n_robot_actions,
            map_size=self.map_size
        )
