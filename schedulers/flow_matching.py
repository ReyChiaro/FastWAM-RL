import math

import torch

from dataclasses import dataclass, field


@dataclass
class FlowTransition:

    sample: torch.Tensor
    log_prob: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor


@dataclass(frozen=True)
class FlowMatchingScheduler:

    shift: float = 5.0
    eps: float = 1e-6
    num_training_steps: int = 1000
    weight_min: float = math.exp(-0.5)
    weight_norm: float = field(init=False)

    def __post_init__(self) -> None:
        # The training distribution is sampled by drawing u ~ Uniform(0, 1)
        # and shifting u into sigma space.  Normalize on that same distribution
        # instead of on the current (arbitrary-sized) training batch.
        u = torch.linspace(0, 1, 4097, dtype=torch.float64)
        sigma = self.shift_sigma(u)
        weight = torch.exp(-2 * (sigma - 0.5).square()) - self.weight_min
        object.__setattr__(self, "weight_norm", float(torch.trapezoid(weight, u).item()))

    def shift_sigma(self, u: torch.Tensor) -> torch.Tensor:
        return self.shift * u / (1.0 + (self.shift - 1) * u)

    def sample_training_sigmas(
        self, batch_size: int, device: torch.device, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        u = torch.rand(batch_size, device=device, dtype=torch.float32, generator=generator)
        return self.shift_sigma(u)

    def sample_inference_sigmas(self, num_inference_steps: int, device: torch.device) -> torch.Tensor:
        ts = torch.linspace(1, 0, num_inference_steps + 1, device=device, dtype=torch.float32)
        return self.shift_sigma(ts)

    def convert_to_model_timesteps(self, sigmas: torch.Tensor) -> torch.Tensor:
        dtype = sigmas.dtype
        return (sigmas * float(self.num_training_steps)).to(dtype=dtype)

    def add_noise(self, xt: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        dtype = xt.dtype
        sigma = sigma.float()
        while sigma.ndim < xt.ndim:
            sigma = sigma.unsqueeze(-1)
        xt = (1.0 - sigma) * xt.float() + sigma * noise.float()
        return xt.to(dtype=dtype)

    def training_target(self, original_sample: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return noise - original_sample

    def training_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.to(dtype=torch.float32)
        weight = torch.exp(-2 * (sigma - 0.5) ** 2) - self.weight_min
        return weight / (self.weight_norm + self.eps)

    def step(self, xt: torch.Tensor, v: torch.Tensor, sigma: torch.Tensor, next_sigma: torch.Tensor) -> torch.Tensor:
        dtype = xt.dtype
        delta_sigma = (next_sigma - sigma).float()
        while delta_sigma.ndim < xt.ndim:
            delta_sigma = delta_sigma.unsqueeze(-1)
        xt = xt.float() + delta_sigma * v.float()
        return xt.to(dtype=dtype)

    def stochastic_step(
        self,
        sample: torch.Tensor,
        velocity: torch.Tensor,
        sigma: torch.Tensor,
        next_sigma: torch.Tensor,
        noise_level: float,
        next_sample: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> FlowTransition:
        """One reverse-SDE transition and its per-sample log probability."""

        dtype = sample.dtype
        xt, vt = sample.float(), velocity.float()
        sigma, next_sigma = sigma.float(), next_sigma.float()

        while sigma.ndim < sample.ndim:
            sigma = sigma.unsqueeze(-1)
            next_sigma = next_sigma.unsqueeze(-1)
        delta = next_sigma - sigma

        # sigma=1 is the pure-noise endpoint.  The reverse-SDE coefficient is
        # singular there, so the first transition uses the next schedule value
        # as the effective endpoint, matching the Flow-GRPO sampler derivation.
        effective_sigma = torch.where(sigma >= 1.0 - self.eps, next_sigma, sigma)
        denominator = (1.0 - effective_sigma).clamp_min(self.eps)
        diffusion = torch.sqrt(sigma / denominator) * noise_level

        mean = xt * (1 + diffusion.square() / (2 * sigma.clamp_min(self.eps)) * delta)
        mean = mean + vt * (1 + diffusion.square() * (1 - sigma) / (2 * sigma.clamp_min(self.eps))) * delta
        std = diffusion * torch.sqrt((-delta).clamp_min(self.eps))

        if next_sample is None:
            noise = torch.randn(sample.shape, device=sample.device, dtype=torch.float32, generator=generator)
            next_xt = mean + std * noise
        else:
            next_xt = next_sample.float()

        log_prob = -0.5 * ((next_xt.detach() - mean) / std).square()
        log_prob = log_prob - torch.log(std) - 0.5 * math.log(2 * math.pi)
        log_prob = log_prob.flatten(1).mean(dim=1)

        return FlowTransition(sample=next_xt.to(dtype), log_prob=log_prob, mean=mean, std=std)
