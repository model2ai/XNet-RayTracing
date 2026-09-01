#Define como treinar a XNet

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.preprocessing import DataTransformer
from utils.xnet_model import XNet


@dataclass
class MappedLoss:
    total_loss: float
    data_loss: Optional[float] = None
    physics_loss: Optional[float] = None
    reg_loss: Optional[float] = None

    def __init__(self, total_loss, data_loss=None, physics_loss=None, reg_loss=None):
        self.total_loss = total_loss.item()
        self.data_loss = data_loss.item() if data_loss is not None else None
        self.physics_loss = physics_loss.item() if physics_loss is not None else None
        self.reg_loss = reg_loss.item() if reg_loss is not None else None


class ArchitectureXNet(object):
    """Training architecture dedicated to XNet and compatible with PINN losses."""

    def __init__(
        self,
        model: XNet,
        loss_fn: callable,
        partial_optimizer: optim.Optimizer,
        physics_fn: Optional[callable] = None,
        partial_scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
        use_weighted_pi: bool = False,
        lambda_physics: float = 1.0,
        data_pipeline: Optional[List[DataTransformer]] = None,
        device=None,
        **_,
    ):
        super().__init__()

        self.physics_loss_weight = lambda_physics
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.physics_fn = physics_fn
        self.use_weighted_pi = use_weighted_pi

        self.partial_optimizer = partial_optimizer
        self.partial_scheduler = partial_scheduler
        self.scheduler = None
        self.optimizer = None
        self.init_optimizer_and_scheduler()

        self.clipping = None
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        self.train_loader = None
        self.val_loader = None

        self.early_stopping = False

        self.losses = []
        self.p_losses = []
        self.d_losses = []
        self.reg_losses = []
        self.val_losses = []
        self.total_epochs = 0

        self.data_pipeline = data_pipeline if data_pipeline is not None else []
        if self.data_pipeline:
            assert all(isinstance(obj, DataTransformer) for obj in self.data_pipeline), (
                "All objects in data_pipeline must be instances of DataTransformer"
            )

        self.verbose = False
        self.warning_flag = False

    def init_optimizer_and_scheduler(self):
        self.optimizer = self.partial_optimizer(self.model.parameters())
        if self.partial_scheduler is not None:
            self.scheduler = self.partial_scheduler(self.optimizer)
        return self.optimizer

    def to(self, device):
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Couldn't send it to {device}, sending it to {self.device} instead.")
            self.model.to(self.device)

    def set_loaders(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def set_clip_backprop(self, clip_value):
        if self.clipping is None:
            self.clipping = []
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                handle = parameter.register_hook(
                    lambda grad: torch.clamp(grad, -clip_value, clip_value)
                )
                self.clipping.append(handle)

    def remove_clip(self):
        if isinstance(self.clipping, list):
            for handle in self.clipping:
                handle.remove()
        self.clipping = None

    def set_early_stopping(self, patience: int = 10):
        self.early_stopping = True
        self.early_stopping_patience = patience
        self.early_stopping_counter = 0
        self.best_val_loss = float("inf")

    def check_early_stopping(self, val_loss: float):
        if self.early_stopping:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

            if self.early_stopping_counter >= self.early_stopping_patience:
                return True
        return False

    def _input_features(self):
        return getattr(self.model, "input_dim", None)

    @property
    def train_step_fn(self) -> callable:
        def perform_train_step_fn(x, y, step: Optional[int] = None) -> torch.Tensor:
            self.model.train()

            input_features = self._input_features()
            model_input = x[:, :input_features] if input_features is not None else x
            if input_features is not None and x.shape[1] != input_features and not self.warning_flag:
                self.warning_flag = True
                print(
                    "[WARNING] Input features are different from the model's input features. "
                    f"Only the first {input_features} features will be used as model input."
                )

            yhat = self.model(model_input)
            data_loss = self.loss_fn(yhat, y)

            physics_loss = None
            if self.physics_fn is not None:
                physics_loss = self.physics_fn(self.model, x)

            loss = data_loss
            if physics_loss is not None:
                loss = loss + self.physics_loss_weight * physics_loss

            loss.backward()
            if callable(self.clipping):
                self.clipping()

            self.optimizer.step()
            self.optimizer.zero_grad()

            return torch.tensor(
                [
                    loss.item(),
                    data_loss.item(),
                    physics_loss.item() if physics_loss is not None else 0.0,
                    0.0,
                ],
                device=self.device,
            )

        return perform_train_step_fn

    @property
    def val_step_fn(self) -> callable:
        def perform_val_step_fn(x, y, step: Optional[int] = None):
            self.model.eval()

            input_features = self._input_features()
            model_input = x[:, :input_features] if input_features is not None else x
            yhat = self.model(model_input)
            data_loss = self.loss_fn(yhat, y)

            return torch.tensor(
                [data_loss.item(), data_loss.item(), 0.0, 0.0],
                device=self.device,
            )

        return perform_val_step_fn

    def _mini_batch(self, validation=False) -> Optional[MappedLoss]:
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn

        if data_loader is None:
            return None

        mini_batch_losses = []
        for step, (x_batch, y_batch) in enumerate(data_loader):
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            mini_batch_losses.append(step_fn(x_batch, y_batch, step))

        losses = torch.stack(mini_batch_losses)
        mean_losses = torch.mean(losses, dim=0)
        if mean_losses.dim() == 0:
            mean_losses = mean_losses.unsqueeze(0)

        if validation and self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(mean_losses[0].item())
            else:
                self.scheduler.step()

        return MappedLoss(*mean_losses)

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def train(self, n_epochs, seed=42, verbose=False, **_):
        self.verbose = verbose
        self.set_seed(seed)

        total_epochs = self.total_epochs + n_epochs
        for epoch in (
            pbar := tqdm(
                range(self.total_epochs, total_epochs),
                total=total_epochs,
                initial=self.total_epochs,
                desc="Training XNet",
            )
        ):
            self.total_epochs += 1

            loss = self._mini_batch(validation=False)
            self.losses.append(loss.total_loss)
            self.d_losses.append(loss.data_loss)
            self.p_losses.append(loss.physics_loss)
            self.reg_losses.append(loss.reg_loss)

            val_loss = self._mini_batch(validation=True)
            if val_loss is not None:
                self.val_losses.append(val_loss.total_loss)

            physics_loss = None
            if loss.physics_loss is not None:
                physics_loss = loss.physics_loss * self.physics_loss_weight

            pbar.set_postfix(
                loss=loss.total_loss,
                data_loss=loss.data_loss,
                physics_loss=physics_loss,
                val_loss=val_loss.total_loss if val_loss is not None else None,
            )

            if val_loss is not None and self.check_early_stopping(val_loss.total_loss):
                print(f"[INFO] Early stopping at epoch {epoch + 1}.")
                break

    def save_checkpoint(self, filename: Path, create_dirs=True):
        if not isinstance(filename, Path):
            filename = Path(filename)

        if create_dirs:
            filename.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": self.total_epochs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": self.losses,
            "d_loss": self.d_losses,
            "p_loss": self.p_losses,
            "reg_losses": self.reg_losses,
            "val_losses": self.val_losses,
        }

        torch.save(checkpoint, filename)
        print(f"Succesfully saved Checkpoint to '{filename}'")

    def load_checkpoint(self, filename: Path):
        if not isinstance(filename, Path):
            filename = Path(filename)

        if not filename.exists():
            print(f"WARNING: Checkpoint file not found at '{filename}'. Model was not loaded.")
            return

        checkpoint = torch.load(filename, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.total_epochs = checkpoint["epoch"]
        self.losses = checkpoint.get("loss", [])
        self.d_losses = checkpoint.get("d_loss", [])
        self.p_losses = checkpoint.get("p_loss", [])
        self.reg_losses = checkpoint.get("reg_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])

        self.model.train()
        print(f"Checkpoint successfully loaded from '{filename}'")

    def predict(self, x):
        self.model.eval()
        x_tensor = torch.as_tensor(x).float()
        input_features = self._input_features()
        if input_features is not None:
            x_tensor = x_tensor[:, :input_features]
        y_hat_tensor = self.model(x_tensor.to(self.device))
        self.model.train()
        return y_hat_tensor.detach().cpu().numpy()

    def plot_losses(self):
        fig, axs = plt.subplots(3, 1, figsize=(10, 4))
        axs[0].plot(self.losses, label="Training Loss", c="b")
        axs[0].set_yscale("log")
        axs[0].set_xlabel("Epochs")
        axs[0].set_ylabel("Loss")
        axs[0].legend()

        axs[1].plot(self.p_losses, label="Training Physics Loss", c="b")
        axs[1].set_yscale("log")
        axs[1].set_xlabel("Epochs")
        axs[1].set_ylabel("Loss")
        axs[1].legend()

        axs[2].plot(self.d_losses, label="Training Data Loss", c="b")
        axs[2].plot(self.val_losses, label="Validation Data Loss", c="r")
        axs[2].set_yscale("log")
        axs[2].set_xlabel("Epochs")
        axs[2].set_ylabel("Loss")
        axs[2].legend()
        plt.tight_layout()
        return fig
