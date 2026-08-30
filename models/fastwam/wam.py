import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Any, Optional
from pathlib import Path

from models.fastwam.mot import FastWAMMoT, ExpertInputs
from models.fastwam.dit import VideoDiT, ActionDiT
from models.fastwam.vae import WanVideoVAE
from models.fastwam.llm import WanTextEncoder, HuggingfaceTokenizer
from schedulers.flow_matching import FlowMatchingScheduler, FlowTransition
from models.utils import CheckpointModule


@dataclass
class StepOutputs:

    video_loss: torch.Tensor
    action_loss: Optional[torch.Tensor] = None


@dataclass
class ModelInputs:

    video_latents: torch.Tensor
    action_latents: torch.Tensor
    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor
    frame_is_pad: torch.Tensor
    action_is_pad: torch.Tensor
    first_frame_latents: torch.Tensor


@dataclass
class ConditionOutputs:

    embeds: torch.Tensor
    mask: torch.Tensor


@dataclass
class ActionCache:

    key_values: list[dict[str, torch.Tensor]]
    attention_mask: torch.Tensor


class ProprioEncoder(CheckpointModule, nn.Linear):
    """One linear projection of the initial proprioceptive state.

    Kept as its own subclass (instead of a bare ``nn.Linear``) so it can carry
    a ``from_pretrained`` like every other FastWAM submodel — the Diffusers
    per-submodel loading convention.  It is stored under the bundle directory
    ``proprio_encoder/``.
    """


class FastWAM(CheckpointModule, nn.Module):

    def __init__(
        self,
        vae: WanVideoVAE,
        tokenizer: HuggingfaceTokenizer,
        text_encoder: WanTextEncoder,
        video_expert: VideoDiT,
        action_expert: ActionDiT,
        video_scheduler: FlowMatchingScheduler,
        action_scheduler: FlowMatchingScheduler,
        proprio_dim: Optional[int] = None,
        use_gradient_checkpointing: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        # The video/action experts live exclusively inside ``mot.mixtures`` so
        # the state dict has a single registration path (``mot.mixtures.*``),
        # matching the original FastWAM checkpoint naming.  ``video_expert`` /
        # ``action_expert`` are read-only views for callers.
        self.mot: FastWAMMoT = FastWAMMoT(
            video_expert,
            action_expert,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        self.proprio_encoder = ProprioEncoder(proprio_dim, text_encoder.dim) if proprio_dim else None

        self.video_scheduler = video_scheduler
        self.action_scheduler = action_scheduler

        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    @property
    def video_expert(self) -> VideoDiT:
        return self.mot.mixtures["video"]

    @property
    def action_expert(self) -> ActionDiT:
        return self.mot.mixtures["action"]

    @property
    def current_device(self) -> torch.device:
        return next(self.mot.parameters()).device

    @property
    def current_dtype(self) -> torch.dtype:
        return next(self.mot.parameters()).dtype

    def build_shared_attn_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_self_attn_mask(
            video_seq_len, video_tokens_per_frame, device
        )
        mask[video_seq_len:, video_seq_len:] = True
        first_frame_len = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_len] = True
        return mask

    def encode_prompts(self, prompts: list[str] | str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize and encode prompts into cross-attention conditioning.

        Deliberate masking policy: prompts are padded to ``seq_len`` (128) by
        the tokenizer, but the *returned* attention mask is the real token
        mask, so padded positions are excluded from cross-attention entirely.
        This keeps the conditioning signal dense for the short LIBERO task
        descriptions instead of diluting it over 128 mostly-empty slots.
        """
        ids, mask = self.tokenizer(prompts, return_mask=True, add_special_tokens=True)
        ids: torch.Tensor = ids.to(device)
        mask: torch.Tensor = mask.to(device=device, dtype=torch.bool)

        prompt_embeds: torch.Tensor = self.text_encoder(ids, mask)
        prompt_embeds = prompt_embeds.masked_fill(~mask.unsqueeze(-1), 0)
        return prompt_embeds, mask

    def encode_proprios(self, proprios: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = proprios.shape[0]
        if self.proprio_encoder is None:
            proprio_embeds = torch.empty(
                (batch_size, 0, self.text_encoder.dim), device=device, dtype=self.current_dtype
            )
            proprio_embeds_mask = torch.empty((batch_size, 0), device=device, dtype=torch.bool)
            return proprio_embeds, proprio_embeds_mask

        initial_proprio = proprios[:, 0, :] if proprios.ndim == 3 else proprios
        initial_proprio = initial_proprio.to(device=device, dtype=self.current_dtype)
        proprio_embeds = self.proprio_encoder(initial_proprio.unsqueeze(1))
        proprio_embeds_mask = torch.ones((batch_size, 1), device=device, dtype=torch.bool)
        return proprio_embeds, proprio_embeds_mask

    def encode_videos(self, videos: torch.Tensor, device: torch.device) -> torch.Tensor:
        return self.vae.encode(videos=videos.to(device=device, dtype=self.current_dtype), device=device)

    def encode_condition(self, prompts: list[str] | str, proprios: torch.Tensor) -> ConditionOutputs:
        prompt_embeds, prompt_mask = self.encode_prompts(prompts, self.current_device)
        proprio_embeds, proprio_mask = self.encode_proprios(proprios, self.current_device)
        return ConditionOutputs(
            embeds=torch.cat([prompt_embeds, proprio_embeds], dim=1),
            mask=torch.cat([prompt_mask, proprio_mask], dim=1),
        )

    def predict_joint_velocity(
        self,
        video_latents: torch.Tensor,
        action_latents: torch.Tensor,
        video_sigma: torch.Tensor,
        action_sigma: torch.Tensor,
        condition: ConditionOutputs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_info = self.video_expert.preprocess(
            video_tokens=video_latents,
            timestep=self.video_scheduler.convert_to_model_timesteps(video_sigma),
            prompt_embeds=condition.embeds,
            prompt_embeds_mask=condition.mask,
        )
        action_info = self.action_expert.preprocess(
            action_tokens=action_latents,
            timestep=self.action_scheduler.convert_to_model_timesteps(action_sigma),
            prompt_embeds=condition.embeds,
            prompt_embeds_mask=condition.mask,
        )
        attention_mask = self.build_shared_attn_mask(
            video_info.hidden_states.shape[1],
            action_info.hidden_states.shape[1],
            video_info.num_tokens_per_frame,
            video_latents.device,
        )
        hidden_states = self.mot(
            video_inputs=ExpertInputs(
                video_info.hidden_states,
                video_info.freqs,
                video_info.prompt_embeds,
                video_info.prompt_embeds_mask,
                video_info.time_projs,
            ),
            action_inputs=ExpertInputs(
                action_info.hidden_states,
                action_info.freqs,
                action_info.prompt_embeds,
                action_info.prompt_embeds_mask,
                action_info.time_projs,
            ),
            attn_mask=attention_mask,
        )
        video_velocity = self.video_expert.postprocess(
            hidden_states["video"], video_info.time_embeds, video_info.grid_size
        )
        action_velocity = self.action_expert.postprocess(hidden_states["action"])
        return video_velocity, action_velocity

    def prepare_action_cache(
        self,
        first_frame_latents: torch.Tensor,
        action_horizon: int,
        condition: ConditionOutputs,
    ) -> ActionCache:
        zero_sigma = torch.zeros(first_frame_latents.shape[0], device=first_frame_latents.device)
        video_info = self.video_expert.preprocess(
            first_frame_latents,
            self.video_scheduler.convert_to_model_timesteps(zero_sigma),
            condition.embeds,
            condition.mask,
        )
        video_length = video_info.hidden_states.shape[1]
        attention_mask = self.build_shared_attn_mask(
            video_length,
            action_horizon,
            video_info.num_tokens_per_frame,
            first_frame_latents.device,
        )
        key_values = self.mot.prefill_video_cache(
            ExpertInputs(
                video_info.hidden_states,
                video_info.freqs,
                video_info.prompt_embeds,
                video_info.prompt_embeds_mask,
                video_info.time_projs,
            ),
            attention_mask[:video_length, :video_length],
        )
        return ActionCache(key_values, attention_mask)

    def predict_action_velocity(
        self,
        action_latents: torch.Tensor,
        action_sigma: torch.Tensor,
        condition: ConditionOutputs,
        cache: ActionCache,
    ) -> torch.Tensor:
        action_info = self.action_expert.preprocess(
            action_latents,
            self.action_scheduler.convert_to_model_timesteps(action_sigma),
            condition.embeds,
            condition.mask,
        )
        hidden_states = self.mot.forward_with_video_cache(
            ExpertInputs(
                action_info.hidden_states,
                action_info.freqs,
                action_info.prompt_embeds,
                action_info.prompt_embeds_mask,
                action_info.time_projs,
            ),
            cache.key_values,
            cache.attention_mask,
        )
        return self.action_expert.postprocess(hidden_states["action"])

    def action_transition(
        self,
        prompts: list[str],
        first_frame: torch.Tensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
        sigma: torch.Tensor,
        next_sigma: torch.Tensor,
        noise_level: float,
        next_action: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> FlowTransition:
        """Differentiable action transition used by rollout replay and GRPO."""
        device, dtype = self.current_device, self.current_dtype
        first_frame = first_frame.to(device=device, dtype=dtype).unsqueeze(2)

        first_frame_latents = self.encode_videos(first_frame, device)
        condition = self.encode_condition(prompts, proprio)
        cache = self.prepare_action_cache(first_frame_latents, action.shape[1], condition)

        action = action.to(device=device, dtype=dtype)
        sigma = sigma.to(device)
        next_sigma = next_sigma.to(device)

        velocity = self.predict_action_velocity(action, sigma, condition, cache)
        return self.action_scheduler.stochastic_step(
            action,
            velocity,
            sigma,
            next_sigma,
            noise_level,
            next_sample=None if next_action is None else next_action.to(device=device, dtype=dtype),
            generator=generator,
        )

    def prepare_inputs(self, sample: dict[str, list[str] | torch.Tensor]) -> ModelInputs:
        """
        video: [B,3,T,H,W] The video size should be dividisible by 16 and the frames should be T=(F-1)*4+1
        frame_is_pad: [B,T]
        action: [B,T,C]
        action_is_pad: [B,T]
        proprio: [B,C]
        """
        prompts: list[str] = sample["prompt"]
        videos: torch.Tensor = sample["video"]
        actions: torch.Tensor = sample["action"]
        proprios: torch.Tensor = sample["proprio"]
        frame_is_pad: torch.Tensor = sample["frame_is_pad"]
        action_is_pad: torch.Tensor = sample["action_is_pad"]

        device = self.current_device
        condition = self.encode_condition(prompts, proprios)
        video_latents = self.encode_videos(videos, device)
        first_frame_latents = video_latents[:, :, 0:1]

        return ModelInputs(
            video_latents=video_latents,
            action_latents=actions.to(device=device, dtype=self.current_dtype),
            prompt_embeds=condition.embeds,
            prompt_embeds_mask=condition.mask,
            first_frame_latents=first_frame_latents,
            frame_is_pad=frame_is_pad.to(device=device, dtype=torch.bool),
            action_is_pad=action_is_pad.to(device=device, dtype=torch.bool),
        )

    def compute_video_loss(
        self, pred: torch.Tensor, gt: torch.Tensor, sigma: torch.Tensor, frame_is_pad: torch.Tensor
    ) -> torch.Tensor:
        loss_field = F.mse_loss(pred.float(), gt.float(), reduction="none").mean(dim=(1, 3, 4))  # [B,C,T,H,W]->[B,T]

        temporal_factor = self.vae.temporal_downsample_factor
        latent_is_pad = frame_is_pad[:, 1:].view(frame_is_pad.shape[0], -1, temporal_factor).all(dim=-1)
        valid_latents = ~latent_is_pad
        loss = (loss_field * valid_latents).sum(dim=-1) / valid_latents.sum(dim=-1).clamp(min=1.0)

        weight = self.video_scheduler.training_weight(sigma)
        return (weight * loss).mean()

    def compute_action_loss(
        self, pred: torch.Tensor, gt: torch.Tensor, sigma: torch.Tensor, action_is_pad: torch.Tensor
    ) -> torch.Tensor:
        loss_field = F.mse_loss(pred.float(), gt.float(), reduction="none").mean(dim=-1)  # [B,T,C]->[B,T]

        valid_latents = ~action_is_pad
        loss = (loss_field * valid_latents).sum(dim=-1) / valid_latents.sum(dim=-1).clamp(min=1.0)

        weight = self.action_scheduler.training_weight(sigma)
        return (weight * loss).mean()

    def training_step(self, sample: dict[str, str | torch.Tensor]) -> StepOutputs:
        inputs = self.prepare_inputs(sample)
        device = inputs.video_latents.device

        # Video
        video_latents = inputs.video_latents
        batch_size = video_latents.shape[0]
        video_noise = torch.randn_like(video_latents)
        video_sigma = self.video_scheduler.sample_training_sigmas(batch_size, device)
        video_xt = self.video_scheduler.add_noise(video_latents, video_noise, video_sigma)
        video_xt[:, :, 0:1] = inputs.first_frame_latents
        video_gt = self.video_scheduler.training_target(video_latents, video_noise)
        video_gt = video_gt[:, :, 1:]  # The first latent frame is provided as conditioning.

        # Action
        action_latents = inputs.action_latents
        action_noise = torch.randn_like(action_latents)
        action_sigma = self.action_scheduler.sample_training_sigmas(batch_size, device)
        action_xt = self.action_scheduler.add_noise(action_latents, action_noise, action_sigma)
        action_gt = self.action_scheduler.training_target(action_latents, action_noise)

        condition = ConditionOutputs(inputs.prompt_embeds, inputs.prompt_embeds_mask)
        pred_video, pred_action = self.predict_joint_velocity(
            video_xt,
            action_xt,
            video_sigma,
            action_sigma,
            condition,
        )
        pred_video = pred_video[:, :, 1:]  # The first latent frame is ground truth.

        # Loss
        video_loss = self.compute_video_loss(pred_video, video_gt, video_sigma, inputs.frame_is_pad)
        action_loss = self.compute_action_loss(pred_action, action_gt, action_sigma, inputs.action_is_pad)

        return StepOutputs(video_loss=video_loss, action_loss=action_loss)

    def forward(
        self,
        sample: dict[str, str | torch.Tensor] | None = None,
        operation: str = "sft",
        **kwargs: Any,
    ) -> StepOutputs | FlowTransition:
        if operation == "sft":
            return self.training_step(sample)
        if operation == "action_transition":
            return self.action_transition(**kwargs)
        raise ValueError(f"Unsupported FastWAM operation {operation!r}.")
