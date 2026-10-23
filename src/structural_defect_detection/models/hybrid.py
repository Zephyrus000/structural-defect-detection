"""Sequential CNN -> spatial Swin blocks -> decoder with high-resolution skips."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.swin_transformer import SwinTransformerBlock

from ..config import ModelConfig


class ConvBlock(nn.Module):
    def __init__(self, incoming: int, outgoing: int, stride: int = 1, efficient: bool = False):
        super().__init__()
        if efficient:
            first = nn.Sequential(
                nn.Conv2d(incoming, incoming, 3, stride, 1, groups=incoming, bias=False),
                nn.Conv2d(incoming, outgoing, 1, bias=False),
            )
        else:
            first = nn.Conv2d(incoming, outgoing, 3, stride, 1, bias=False)
        self.layers = nn.Sequential(
            first,
            nn.GroupNorm(math.gcd(outgoing, 8), outgoing),
            nn.GELU(),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(math.gcd(outgoing, 8), outgoing),
            nn.GELU(),
        )

    def forward(self, value):
        return self.layers(value)


@dataclass(frozen=True)
class FeatureInfo:
    name: str
    channels: int
    stride: int


class CNNEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        channels = config.channels[: config.feature_level + 1]
        self.info = tuple(FeatureInfo(f"level{i}", c, 2**i) for i, c in enumerate(channels))
        self.stages = nn.ModuleList(
            [
                ConvBlock(
                    3 if i == 0 else channels[i - 1],
                    c,
                    1 if i == 0 else 2,
                    config.cnn == "efficient",
                )
                for i, c in enumerate(channels)
            ]
        )

    def forward(self, image):
        features = []
        for stage in self.stages:
            image = stage(image)
            features.append(image)
        return features


class FeatureProjection(nn.Module):
    def __init__(self, incoming: int, embed_dim: int):
        super().__init__()
        self.project = nn.Conv2d(incoming, embed_dim, 1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, feature):
        return self.norm(self.project(feature).permute(0, 2, 3, 1))


class SwinFeatureProcessor(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                SwinTransformerBlock(
                    dim=config.embed_dim,
                    num_heads=config.heads,
                    window_size=[config.window_size] * 2,
                    shift_size=[0 if i % 2 == 0 else config.window_size // 2] * 2,
                    dropout=config.dropout,
                    attention_dropout=config.dropout,
                    stochastic_depth_prob=0.0,
                )
                for i in range(config.depth)
            ]
        )

    def forward(self, feature):
        return self.blocks(feature).permute(0, 3, 1, 2).contiguous()


class SegmentationDecoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.blocks = nn.ModuleList()
        incoming = config.embed_dim
        for skip_channels in reversed(config.channels[: config.feature_level]):
            self.blocks.append(ConvBlock(incoming + skip_channels, config.decoder_channels))
            incoming = config.decoder_channels
        self.head = nn.Conv2d(incoming, 1, 1)

    def forward(self, processed, skips, output_size):
        for block, skip in zip(self.blocks, reversed(skips), strict=True):
            processed = F.interpolate(
                processed, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
            processed = block(torch.cat([processed, skip], dim=1))
        logits = self.head(processed)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class HybridCNNSwin(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = CNNEncoder(config)
        self.projection = FeatureProjection(config.channels[config.feature_level], config.embed_dim)
        self.processor = SwinFeatureProcessor(config) if config.name == "hybrid" else None
        self.decoder = SegmentationDecoder(config)
        self.initialization_report = {
            "source": None,
            "loaded": [],
            "reinitialized": list(self.state_dict()),
            "normalization": "CNN GroupNorm; projection/Swin LayerNorm",
            "cnn_only_replacement": "identity on projected features"
            if self.processor is None
            else None,
        }

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected RGB images [B,3,H,W]")
        features = self.encoder(image)
        projected = self.projection(features[-1])
        processed = (
            self.processor(projected)
            if self.processor is not None
            else projected.permute(0, 3, 1, 2)
        )
        return self.decoder(processed, features[:-1], image.shape[-2:])


def build_model(config: ModelConfig, initialize: bool = True) -> HybridCNNSwin:
    model = HybridCNNSwin(config)
    if initialize and config.init_weights:
        state = torch.load(config.init_weights, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not all(
            isinstance(v, torch.Tensor) for v in state.values()
        ):
            raise ValueError("init_weights must be a tensor-only state_dict")
        current = model.state_dict()
        loaded = {k: v for k, v in state.items() if k in current and v.shape == current[k].shape}
        if not loaded:
            raise ValueError("No compatible pretrained layers found")
        model.load_state_dict(loaded, strict=False)
        model.initialization_report.update(
            source=config.init_weights,
            loaded=sorted(loaded),
            reinitialized=sorted(current.keys() - loaded.keys()),
            unused=sorted(state.keys() - loaded.keys()),
        )
    return model
