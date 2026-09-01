"""XNet components used by the Marmousi seismic ray-tracing experiments.

Reference architecture details:
- Network: feed-forward fully connected model stored in ``nn.Sequential``.
- Blocks: first ``Linear(input_dim, hidden_dim)`` plus Cauchy activation,
  followed by ``n_layers - 1`` hidden ``Linear(hidden_dim, hidden_dim)`` plus
  Cauchy activation blocks, and a final linear output layer.
- Activation: parameterized Cauchy function
  ``(lam1 * x + lam2) / (x**2 + d**2 + 1e-9)`` with trainable scalar
  parameters initialized as ``lam1=0``, ``lam2=1``, and ``d=1``.
- Initialization: Xavier uniform for every ``nn.Linear`` weight and zero bias.
- Architectural hyperparameters used in the reference notebook:
  ``input_dim=4``, ``hidden_dim=64``, ``output_dim=4``, ``n_layers=4``.
- Derivatives: the model uses only PyTorch tensor operations and does not
  detach, cast through NumPy, or mutate inputs during forward propagation,
  preserving autograd for physics-informed derivatives.
"""

#Define o que é a XNet

import torch
import torch.nn as nn


class CauchyActivation(nn.Module):
    """Parameterized Cauchy activation used by the XNet reference notebook."""

    def __init__(self):
        super().__init__()
        self.lam1 = nn.Parameter(torch.tensor(0.0))
        self.lam2 = nn.Parameter(torch.tensor(1.0))
        self.d = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return (self.lam1 * x + self.lam2) / (x**2 + self.d**2 + 1e-9)


class XNet(nn.Module):
    """XNet architecture reproduced from xnet_notebook.ipynb.

    Architecture:
        Linear(input_dim, hidden_dim) -> CauchyActivation,
        followed by n_layers - 1 blocks of
        Linear(hidden_dim, hidden_dim) -> CauchyActivation,
        followed by Linear(hidden_dim, output_dim).

    Linear layers use Xavier uniform weight initialization and zero bias.
    The default hidden_dim and n_layers match the values used in the
    reference notebook.
    """

    def __init__(self, input_dim, hidden_dim=64, output_dim=4, n_layers=4):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        layers = [nn.Linear(input_dim, hidden_dim), CauchyActivation()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), CauchyActivation()])
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self.network.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x)
