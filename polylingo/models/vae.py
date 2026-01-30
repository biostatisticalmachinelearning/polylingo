"""Variational Autoencoder architectures."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    """Convolutional VAE for 64x64 grayscale images."""

    def __init__(
        self,
        latent_dim: int = 64,
        image_channels: int = 1,
        hidden_dims: tuple = (32, 64, 128, 256),
        image_size: int = 64,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.image_channels = image_channels
        self.hidden_dims = hidden_dims
        self.image_size = image_size

        # Calculate size after conv layers
        self.final_size = image_size // (2 ** len(hidden_dims))
        self.flat_size = hidden_dims[-1] * self.final_size * self.final_size

        # Encoder
        encoder_layers = []
        in_channels = image_channels
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Conv2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            in_channels = h_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_var = nn.Linear(self.flat_size, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flat_size)

        decoder_layers = []
        reversed_dims = list(reversed(hidden_dims))
        for i in range(len(reversed_dims) - 1):
            decoder_layers.extend([
                nn.ConvTranspose2d(
                    reversed_dims[i], reversed_dims[i + 1],
                    kernel_size=4, stride=2, padding=1
                ),
                nn.BatchNorm2d(reversed_dims[i + 1]),
                nn.LeakyReLU(0.2, inplace=True),
            ])

        decoder_layers.extend([
            nn.ConvTranspose2d(
                reversed_dims[-1], image_channels,
                kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid(),
        ])

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_var(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        h = self.decoder_input(z)
        h = h.view(h.size(0), self.hidden_dims[-1], self.final_size, self.final_size)
        return self.decoder(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Sample from latent space."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation (mean)."""
        mu, _ = self.encode(x)
        return mu


class ConditionalConvVAE(ConvVAE):
    """Conditional VAE that conditions on class labels."""

    def __init__(
        self,
        latent_dim: int = 64,
        image_channels: int = 1,
        hidden_dims: tuple = (32, 64, 128, 256),
        image_size: int = 64,
        num_classes: int = 52,
    ):
        super().__init__(latent_dim, image_channels, hidden_dims, image_size)

        self.num_classes = num_classes
        self.class_embedding = nn.Embedding(num_classes, latent_dim)

        # Override encoder output layers
        self.fc_mu = nn.Linear(self.flat_size + latent_dim, latent_dim)
        self.fc_var = nn.Linear(self.flat_size + latent_dim, latent_dim)

        # Override decoder input
        self.decoder_input = nn.Linear(latent_dim + latent_dim, self.flat_size)

    def encode(
        self, x: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = h.view(h.size(0), -1)

        if labels is not None:
            c = self.class_embedding(labels)
            h = torch.cat([h, c], dim=1)
        else:
            c = torch.zeros(h.size(0), self.latent_dim, device=h.device)
            h = torch.cat([h, c], dim=1)

        return self.fc_mu(h), self.fc_var(h)

    def decode(
        self, z: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if labels is not None:
            c = self.class_embedding(labels)
            z = torch.cat([z, c], dim=1)
        else:
            c = torch.zeros(z.size(0), self.latent_dim, device=z.device)
            z = torch.cat([z, c], dim=1)

        h = self.decoder_input(z)
        h = h.view(h.size(0), self.hidden_dims[-1], self.final_size, self.final_size)
        return self.decoder(h)

    def forward(
        self, x: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x, labels)
        z = self.reparameterize(mu, log_var)
        return self.decode(z, labels), mu, log_var


def vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute VAE loss.

    Args:
        reconstruction: Reconstructed images.
        target: Original images.
        mu: Latent mean.
        log_var: Latent log variance.
        beta: KL divergence weight (beta-VAE).

    Returns:
        Tuple of (total_loss, recon_loss, kl_loss).
    """
    recon_loss = F.binary_cross_entropy(
        reconstruction, target, reduction='sum'
    ) / target.size(0)

    kl_loss = -0.5 * torch.sum(
        1 + log_var - mu.pow(2) - log_var.exp()
    ) / target.size(0)

    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss
