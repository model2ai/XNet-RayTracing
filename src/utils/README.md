# XNet utilities

This package contains the reusable components used by the XNet seismic ray-tracing experiments.

- [`xnet_model.py`](xnet_model.py): XNet and its parameterized Cauchy activation.
- [`architecture_xnet.py`](architecture_xnet.py): XNet training, physics-informed loss integration, checkpointing and prediction.
- [`architecture_mlp.py`](architecture_mlp.py): MLP baseline used in the architecture comparison.
- [`preprocessing.py`](preprocessing.py): tensor data transformations.
- [`metrics.py`](metrics.py): regression metrics used in the paper experiments.
- [`weighted_pi.py`](weighted_pi.py): frequency-based weighting of the physics-informed training samples and interactive diagnostics.
