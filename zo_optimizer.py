"""
zo_optimizer.py — SPSA (Simultaneous Perturbation Stochastic Approximation) optimizer.

Implements efficient zero-order optimization using simultaneous random perturbations
instead of per-parameter finite differences.

Key features:
  - SPSA: Only 2 forward passes per gradient estimate regardless of parameter count
  - Multi-sample averaging: n_samples independent SPSA estimates averaged per step
  - Adam-style moments: Adaptive learning rates per parameter with momentum
  - Cosine epsilon decay: Large perturbations early, small ones late
  - Rademacher perturbations: ±1 discrete distribution for uniform magnitude
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn


class ZeroOrderOptimizer:
    """Zero-order optimizer using Simultaneous Perturbation Stochastic Approximation.
    
    Instead of computing finite differences for each parameter independently (requiring
    2d forward passes where d is parameter count), SPSA uses a single random direction
    u to perturb all parameters simultaneously:
    
        grad ≈ (f(x + ε·u) - f(x - ε·u)) / (2ε) × u
    
    This requires only 2 forward passes regardless of parameter count.
    
    Args:
        model: The nn.Module to optimize
        lr: Base learning rate (will be adapted by Adam moments)
        eps: Initial perturbation magnitude
        n_samples: Number of independent SPSA estimates to average per step (reduces noise)
        beta1: Adam first moment decay rate (momentum)
        beta2: Adam second moment decay rate (for adaptive lr)
        total_steps: Total optimization steps (used for cosine epsilon decay schedule)
        perturbation_mode: "rademacher" (±1) or "gaussian" (N(0,1))
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-6,
        eps: float = 5e-3,
        n_samples: int = 32,
        beta1: float = 0.9,
        beta2: float = 0.999,
        total_steps: int = 128,
        perturbation_mode: str = "rademacher",
    ) -> None:
        self.model = model
        self.lr = lr
        self.eps_init = eps
        self.n_samples = n_samples
        self.beta1 = beta1
        self.beta2 = beta2
        self.total_steps = total_steps
        self.step_count = 0

        if perturbation_mode not in ("rademacher", "gaussian"):
            raise ValueError(
                f"perturbation_mode must be 'rademacher' or 'gaussian', "
                f"got '{perturbation_mode}'"
            )
        self.perturbation_mode = perturbation_mode

        # Layers to optimize (default: final classification head)
        self.layer_names: list[str] = ["fc.weight", "fc.bias"]

        # Adam moment tracking
        self.m: dict[str, torch.Tensor] = {}  # First moment (momentum)
        self.v: dict[str, torch.Tensor] = {}  # Second moment (variance)

    def _get_active_params(self) -> dict[str, nn.Parameter]:
        """Get all active parameters."""
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(
                f"Layer names not found in model: {missing}. "
                f"Available: {list(named.keys())}"
            )
        return {n: named[n] for n in self.layer_names}

    def _sample_perturbation(self, param: torch.Tensor) -> torch.Tensor:
        """Sample perturbation u with the chosen mode."""
        if self.perturbation_mode == "rademacher":
            # Discrete ±1 distribution
            return torch.randint(0, 2, param.shape, device=param.device) * 2.0 - 1.0
        else:  # gaussian
            return torch.randn_like(param)

    def _get_current_eps(self) -> float:
        """Compute cosine-decayed perturbation magnitude."""
        # Cosine annealing: starts at eps_init, ends at 0.1 * eps_init
        t = self.step_count / max(self.total_steps, 1)
        eps = self.eps_init * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * t)))
        return eps

    def _estimate_grad_spsa(
        self, loss_fn: Callable[[], float], params: dict[str, nn.Parameter]
    ) -> dict[str, torch.Tensor]:
        """Estimate gradient using SPSA with multi-sample averaging."""
        
        eps = self._get_current_eps()
        grad_estimates = {name: torch.zeros_like(p) for name, p in params.items()}

        # Average n_samples independent SPSA estimates
        for _ in range(self.n_samples):
            # Sample random perturbation direction
            u = {name: self._sample_perturbation(p) for name, p in params.items()}

            # Forward pass: f(x + ε·u)
            with torch.no_grad():
                for name, param in params.items():
                    param.data.add_(eps * u[name])
            f_plus = loss_fn()

            # Backward pass: f(x - ε·u)
            with torch.no_grad():
                for name, param in params.items():
                    param.data.sub_(2.0 * eps * u[name])
            f_minus = loss_fn()

            # Restore parameters
            with torch.no_grad():
                for name, param in params.items():
                    param.data.add_(eps * u[name])

            # SPSA gradient: (f+ - f-) / (2ε) × u
            grad_coeff = (f_plus - f_minus) / (2.0 * eps)

            for name, param in params.items():
                grad_estimates[name] += grad_coeff * u[name]

        # Average over samples
        for name in grad_estimates:
            grad_estimates[name] /= self.n_samples

        return grad_estimates

    def _update_params_adam(
        self,
        params: dict[str, nn.Parameter],
        grads: dict[str, torch.Tensor],
    ) -> None:
        """Apply Adam-style update with momentum and adaptive learning rates."""
        
        # Initialize moments on first step
        if not self.m:
            self.m = {name: torch.zeros_like(p) for name, p in params.items()}
            self.v = {name: torch.zeros_like(p) for name, p in params.items()}

        with torch.no_grad():
            # Bias correction factors
            bias_correction1 = 1.0 - self.beta1 ** (self.step_count + 1)
            bias_correction2 = 1.0 - self.beta2 ** (self.step_count + 1)

            for name, param in params.items():
                grad = grads[name]

                # Update biased first moment (momentum)
                self.m[name].mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)

                # Update biased second moment (variance)
                self.v[name].mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)

                # Bias-corrected moments
                m_hat = self.m[name] / bias_correction1
                v_hat = self.v[name] / bias_correction2

                # Adam update
                update = self.lr * m_hat / (torch.sqrt(v_hat) + 1e-8)
                param.data.sub_(update)

    def step(self, loss_fn: Callable[[], float]) -> float:
        """Perform one SPSA optimization step.
        
        Args:
            loss_fn: Callable returning scalar loss on current batch
            
        Returns:
            Loss value at the start of the step
        """
        params = self._get_active_params()

        # Record initial loss
        with torch.no_grad():
            loss_initial = loss_fn()

        # Estimate gradient via SPSA
        grads = self._estimate_grad_spsa(loss_fn, params)

        # Apply Adam update
        self._update_params_adam(params, grads)

        self.step_count += 1

        return float(loss_initial)
