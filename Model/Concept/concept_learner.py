import torch
import torch.nn as nn
from typing import Type


class ConceptLearner(nn.Module):
    def __init__(
        self,
        in_D: int,
        out_D: int,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        activation: Type[nn.Module] = nn.ReLU
    ):
        """
        MLP that takes two inputs of size in_D and outputs a tensor of size out_D.
        Args:
            in_D (int): Input dimension (for each input tensor).
            out_D (int): Output dimension.
            hidden_dim (int | None): Hidden layer dimension. Default is 2 * in_D.
            num_layers (int): Number of linear layers (≥ 2). Default is 2.
            activation (Type[nn.Module]): Activation function class (not instance). Default is nn.ReLU.
        """
        super().__init__()
        hidden_dim = hidden_dim or 2 * in_D
        layers: list[nn.Module] = []
        in_dim = 2 * in_D  # concatenate two in_D-sized inputs
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(activation())
        layers.append(nn.Linear(hidden_dim, out_D))
        self.net = nn.Sequential(*layers)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x1 (torch.Tensor): First input tensor of shape (..., in_D)
            x2 (torch.Tensor): Second input tensor of shape (..., in_D)
        Returns:
            torch.Tensor: Output tensor of shape (..., out_D)
        """
        x = torch.cat([x1, x2], dim=-1)
        return self.net(x)
