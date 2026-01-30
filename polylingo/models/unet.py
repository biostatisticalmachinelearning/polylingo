"""U-Net architecture for diffusion model denoising."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class AttentionBlock(nn.Module):
    """Self-attention block."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)

        qkv = self.qkv(h)
        q, k, v = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W).unbind(1)

        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum("bhcn,bhcm->bhnm", q, k) * scale
        attn = attn.softmax(dim=-1)

        h = torch.einsum("bhnm,bhcm->bhcn", attn, v)
        h = h.reshape(B, C, H, W)
        h = self.proj(h)

        return x + h


class SimpleUNet(nn.Module):
    """Simplified U-Net for diffusion denoising."""

    def __init__(
        self,
        image_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4, 8),
        dropout: float = 0.1,
        num_classes: Optional[int] = None,
    ):
        super().__init__()

        self.num_classes = num_classes
        channels = [base_channels * m for m in channel_mults]

        # Time embedding
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Class embedding
        if num_classes is not None:
            self.class_emb = nn.Embedding(num_classes + 1, time_dim)
            self.null_class = num_classes
        else:
            self.class_emb = None

        # Encoder
        self.enc_conv_in = nn.Conv2d(image_channels, channels[0], 3, padding=1)

        self.enc_blocks = nn.ModuleList()
        self.enc_downs = nn.ModuleList()

        for i in range(len(channels) - 1):
            self.enc_blocks.append(self._make_block(channels[i], channels[i], time_dim, dropout))
            self.enc_blocks.append(self._make_block(channels[i], channels[i], time_dim, dropout))
            self.enc_downs.append(nn.Conv2d(channels[i], channels[i + 1], 4, stride=2, padding=1))

        # Bottleneck
        self.mid_block1 = self._make_block(channels[-1], channels[-1], time_dim, dropout)
        self.mid_attn = AttentionBlock(channels[-1])
        self.mid_block2 = self._make_block(channels[-1], channels[-1], time_dim, dropout)

        # Decoder
        self.dec_blocks = nn.ModuleList()
        self.dec_ups = nn.ModuleList()

        for i in range(len(channels) - 1, 0, -1):
            self.dec_ups.append(nn.ConvTranspose2d(channels[i], channels[i - 1], 4, stride=2, padding=1))
            self.dec_blocks.append(self._make_block(channels[i - 1] * 2, channels[i - 1], time_dim, dropout))
            self.dec_blocks.append(self._make_block(channels[i - 1], channels[i - 1], time_dim, dropout))

        self.dec_conv_out = nn.Sequential(
            nn.GroupNorm(32, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], image_channels, 3, padding=1),
        )

    def _make_block(self, in_ch: int, out_ch: int, time_dim: int, dropout: float):
        return nn.ModuleDict({
            'norm1': nn.GroupNorm(32, in_ch),
            'conv1': nn.Conv2d(in_ch, out_ch, 3, padding=1),
            'time_proj': nn.Linear(time_dim, out_ch),
            'norm2': nn.GroupNorm(32, out_ch),
            'conv2': nn.Conv2d(out_ch, out_ch, 3, padding=1),
            'dropout': nn.Dropout(dropout),
            'skip': nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity(),
        })

    def _apply_block(self, block, x, t_emb):
        h = block['norm1'](x)
        h = F.silu(h)
        h = block['conv1'](h)
        h = h + block['time_proj'](t_emb)[:, :, None, None]
        h = block['norm2'](h)
        h = F.silu(h)
        h = block['dropout'](h)
        h = block['conv2'](h)
        return h + block['skip'](x)

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Time embedding
        t_emb = self.time_mlp(time)

        # Add class embedding
        if self.class_emb is not None:
            if class_labels is None:
                class_labels = torch.full((x.size(0),), self.null_class, device=x.device, dtype=torch.long)
            t_emb = t_emb + self.class_emb(class_labels)

        # Encoder
        h = self.enc_conv_in(x)
        skips = []

        block_idx = 0
        for i, down in enumerate(self.enc_downs):
            h = self._apply_block(self.enc_blocks[block_idx], h, t_emb)
            block_idx += 1
            h = self._apply_block(self.enc_blocks[block_idx], h, t_emb)
            block_idx += 1
            skips.append(h)
            h = down(h)

        # Bottleneck
        h = self._apply_block(self.mid_block1, h, t_emb)
        h = self.mid_attn(h)
        h = self._apply_block(self.mid_block2, h, t_emb)

        # Decoder
        block_idx = 0
        for i, up in enumerate(self.dec_ups):
            h = up(h)
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = self._apply_block(self.dec_blocks[block_idx], h, t_emb)
            block_idx += 1
            h = self._apply_block(self.dec_blocks[block_idx], h, t_emb)
            block_idx += 1

        return self.dec_conv_out(h)
