"""Gaussian Diffusion process implementation."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Linear beta schedule."""
    return torch.linspace(beta_start, beta_end, timesteps)


class GaussianDiffusion(nn.Module):
    """Gaussian Diffusion process for training and sampling."""

    def __init__(
        self,
        model: nn.Module,
        image_size: int = 64,
        image_channels: int = 1,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()

        self.model = model
        self.image_size = image_size
        self.image_channels = image_channels
        self.num_timesteps = timesteps

        # Define beta schedule
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)

        # Pre-compute diffusion parameters
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register as buffers (not parameters, but should be moved with model)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """Extract coefficients at specified timesteps."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward diffusion process: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise."""
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute posterior q(x_{t-1} | x_t, x_0)."""
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance

    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute predicted mean and variance for p(x_{t-1} | x_t)."""
        # Predict noise
        if cfg_scale > 1.0 and class_labels is not None:
            # Classifier-free guidance
            noise_cond = self.model(x_t, t, class_labels)
            noise_uncond = self.model(x_t, t, None)
            noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = self.model(x_t, t, class_labels)

        # Predict x_0
        x_start = self.predict_start_from_noise(x_t, t, noise_pred)
        x_start = torch.clamp(x_start, -1, 1)

        # Compute posterior
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start, x_t, t)

        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """Sample x_{t-1} from p(x_{t-1} | x_t)."""
        model_mean, _, model_log_variance = self.p_mean_variance(x_t, t, class_labels, cfg_scale)

        noise = torch.randn_like(x_t)
        # No noise at t=0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))

        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        class_labels: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        device: Optional[torch.device] = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Generate samples by running the reverse diffusion process."""
        if device is None:
            device = next(self.model.parameters()).device

        # Start from pure noise
        shape = (batch_size, self.image_channels, self.image_size, self.image_size)
        x = torch.randn(shape, device=device)

        # Reverse diffusion
        timesteps = list(reversed(range(self.num_timesteps)))
        if show_progress:
            timesteps = tqdm(timesteps, desc="Sampling")

        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch, class_labels, cfg_scale)

        return x

    @torch.no_grad()
    def sample_ddim(
        self,
        batch_size: int,
        class_labels: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        ddim_steps: int = 50,
        eta: float = 0.0,
        device: Optional[torch.device] = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """DDIM sampling for faster generation."""
        if device is None:
            device = next(self.model.parameters()).device

        # Create timestep sequence
        step_size = self.num_timesteps // ddim_steps
        timesteps = list(range(0, self.num_timesteps, step_size))[::-1]

        shape = (batch_size, self.image_channels, self.image_size, self.image_size)
        x = torch.randn(shape, device=device)

        if show_progress:
            timesteps = tqdm(timesteps, desc="DDIM Sampling")

        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Predict noise
            if cfg_scale > 1.0 and class_labels is not None:
                noise_cond = self.model(x, t_batch, class_labels)
                noise_uncond = self.model(x, t_batch, None)
                noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = self.model(x, t_batch, class_labels)

            # Predict x_0
            x_start = self.predict_start_from_noise(x, t_batch, noise_pred)
            x_start = torch.clamp(x_start, -1, 1)

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_t = self.alphas_cumprod[t]
                alpha_t_prev = self.alphas_cumprod[t_prev]

                sigma = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))

                # Direction pointing to x_t
                pred_dir = torch.sqrt(1 - alpha_t_prev - sigma ** 2) * noise_pred

                # Random noise
                noise = sigma * torch.randn_like(x) if sigma > 0 else 0

                x = torch.sqrt(alpha_t_prev) * x_start + pred_dir + noise
            else:
                x = x_start

        return x

    def training_loss(
        self,
        x_start: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute training loss (simplified MSE loss on noise prediction)."""
        batch_size = x_start.size(0)
        device = x_start.device

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long)

        # Sample noise
        if noise is None:
            noise = torch.randn_like(x_start)

        # Get noisy image
        x_noisy = self.q_sample(x_start, t, noise)

        # Predict noise
        noise_pred = self.model(x_noisy, t, class_labels)

        # MSE loss
        loss = F.mse_loss(noise_pred, noise)

        return loss
