"""U-Net model for diffusion denoising."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


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


class ResidualBlock(nn.Module):
    """Residual block with time and class conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        class_emb_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2),
        )

        self.class_mlp = None
        if class_emb_dim is not None:
            self.class_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(class_emb_dim, out_channels * 2),
            )

        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        class_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.conv1(x)

        # Add time embedding (scale and shift)
        time_emb = self.time_mlp(time_emb)
        time_scale, time_shift = time_emb.chunk(2, dim=1)
        h = h * (1 + time_scale[:, :, None, None]) + time_shift[:, :, None, None]

        # Add class embedding if provided
        if class_emb is not None and self.class_mlp is not None:
            class_emb = self.class_mlp(class_emb)
            class_scale, class_shift = class_emb.chunk(2, dim=1)
            h = h * (1 + class_scale[:, :, None, None]) + class_shift[:, :, None, None]

        h = self.conv2(h)
        return h + self.shortcut(x)


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

        # Attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum("bhcn,bhcm->bhnm", q, k) * scale
        attn = attn.softmax(dim=-1)

        h = torch.einsum("bhnm,bhcm->bhcn", attn, v)
        h = h.reshape(B, C, H, W)
        h = self.proj(h)

        return x + h


class Downsample(nn.Module):
    """Downsample by factor of 2."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsample by factor of 2."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """U-Net for diffusion model denoising."""

    def __init__(
        self,
        image_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (16, 8),
        dropout: float = 0.1,
        num_classes: Optional[int] = None,
        image_size: int = 64,
    ):
        super().__init__()

        self.image_channels = image_channels
        self.base_channels = base_channels
        self.num_classes = num_classes

        # Time embedding
        time_emb_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Class embedding (for conditional generation)
        class_emb_dim = None
        if num_classes is not None:
            class_emb_dim = base_channels * 4
            self.class_embedding = nn.Embedding(num_classes + 1, class_emb_dim)  # +1 for unconditional
            self.null_class = num_classes  # Use last index for unconditional

        # Initial convolution
        self.init_conv = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        channels = [base_channels]
        current_channels = base_channels
        current_resolution = image_size

        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResidualBlock(
                        current_channels, out_channels, time_emb_dim, class_emb_dim, dropout
                    )
                )
                current_channels = out_channels
                channels.append(current_channels)

                # Add attention at specified resolutions
                if current_resolution in attention_resolutions:
                    self.down_blocks.append(AttentionBlock(current_channels))
                    channels.append(current_channels)

            # Downsample (except last level)
            if i < len(channel_mults) - 1:
                self.down_samples.append(Downsample(current_channels))
                channels.append(current_channels)
                current_resolution //= 2

        # Middle blocks
        self.mid_block1 = ResidualBlock(
            current_channels, current_channels, time_emb_dim, class_emb_dim, dropout
        )
        self.mid_attention = AttentionBlock(current_channels)
        self.mid_block2 = ResidualBlock(
            current_channels, current_channels, time_emb_dim, class_emb_dim, dropout
        )

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for i, mult in enumerate(reversed(channel_mults)):
            out_channels = base_channels * mult

            for j in range(num_res_blocks + 1):
                skip_channels = channels.pop()
                self.up_blocks.append(
                    ResidualBlock(
                        current_channels + skip_channels,
                        out_channels,
                        time_emb_dim,
                        class_emb_dim,
                        dropout,
                    )
                )
                current_channels = out_channels

                # Add attention at specified resolutions
                if current_resolution in attention_resolutions and j == num_res_blocks:
                    self.up_blocks.append(AttentionBlock(current_channels))

            # Upsample (except last level)
            if i < len(channel_mults) - 1:
                self.up_samples.append(Upsample(current_channels))
                current_resolution *= 2

        # Output
        self.out_norm = nn.GroupNorm(32, current_channels)
        self.out_conv = nn.Conv2d(current_channels, image_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy images [B, C, H, W]
            time: Timesteps [B]
            class_labels: Class labels [B] (optional, for conditional generation)

        Returns:
            Predicted noise [B, C, H, W]
        """
        # Time embedding
        time_emb = self.time_mlp(time)

        # Class embedding
        class_emb = None
        if self.num_classes is not None:
            if class_labels is None:
                # Use null class for unconditional
                class_labels = torch.full(
                    (x.size(0),), self.null_class, device=x.device, dtype=torch.long
                )
            class_emb = self.class_embedding(class_labels)

        # Initial conv
        h = self.init_conv(x)
        skip_connections = [h]

        # Downsampling
        down_sample_idx = 0
        for block in self.down_blocks:
            if isinstance(block, ResidualBlock):
                h = block(h, time_emb, class_emb)
            else:
                h = block(h)
            skip_connections.append(h)

            # Check if we need to downsample
            if down_sample_idx < len(self.down_samples):
                if len(skip_connections) > 0 and (
                    len(skip_connections) - 1
                ) % (2 if any(s in [16, 8] for s in [h.size(-1)]) else 1) == 0:
                    pass  # Attention block, don't downsample yet

        # Apply downsamples at the right places
        skip_connections = [h]
        h = self.init_conv(x)
        skip_connections = [h]

        block_idx = 0
        sample_idx = 0
        blocks_per_level = []

        for i, mult in enumerate(self.channel_mults if hasattr(self, 'channel_mults') else (1, 2, 4, 8)):
            level_blocks = 0
            for block in self.down_blocks[block_idx:]:
                if isinstance(block, ResidualBlock):
                    h = block(h, time_emb, class_emb)
                    level_blocks += 1
                else:  # AttentionBlock
                    h = block(h)
                skip_connections.append(h)
                block_idx += 1

                if level_blocks >= 2:  # num_res_blocks
                    break

            if sample_idx < len(self.down_samples):
                h = self.down_samples[sample_idx](h)
                skip_connections.append(h)
                sample_idx += 1

        # Middle
        h = self.mid_block1(h, time_emb, class_emb)
        h = self.mid_attention(h)
        h = self.mid_block2(h, time_emb, class_emb)

        # Upsampling
        sample_idx = 0
        for block in self.up_blocks:
            if isinstance(block, ResidualBlock):
                skip = skip_connections.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, time_emb, class_emb)
            else:  # AttentionBlock
                h = block(h)

            # Check if we need to upsample
            if sample_idx < len(self.up_samples) and len(skip_connections) > 0:
                if skip_connections[-1].size(-1) > h.size(-1):
                    h = self.up_samples[sample_idx](h)
                    sample_idx += 1

        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        return self.out_conv(h)


class SimpleUNet(nn.Module):
    """Simplified U-Net for diffusion - more straightforward implementation."""

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
            self.class_emb = nn.Embedding(num_classes + 1, time_dim)  # +1 for null
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
