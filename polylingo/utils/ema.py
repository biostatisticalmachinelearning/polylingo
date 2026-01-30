"""Exponential Moving Average for model parameters."""

import torch.nn as nn


class EMA:
    """Exponential Moving Average of model parameters.

    Maintains shadow copies of model parameters that are updated as
    an exponential moving average during training. This can improve
    generation quality for diffusion models.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """Initialize EMA.

        Args:
            model: The model whose parameters to track.
            decay: EMA decay rate. Higher values give more weight to history.
        """
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        """Update shadow parameters with current model parameters.

        Args:
            model: The model with updated parameters.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply(self, model: nn.Module):
        """Apply shadow parameters to the model.

        Backs up current parameters before applying. Use restore() to undo.

        Args:
            model: The model to apply shadow parameters to.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model: nn.Module):
        """Restore original parameters after apply().

        Args:
            model: The model to restore parameters for.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self) -> dict:
        """Get EMA state for checkpointing."""
        return {
            "decay": self.decay,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state_dict: dict):
        """Load EMA state from checkpoint."""
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]
