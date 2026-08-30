# Training entrypoint
# - hydra-driven configurations


import hydra
import torch

from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import OmegaConf

from models import FastWAM
from data.data_module import DataModule
from trainers.base import BaseTrainer
from evaluation.evaluator import Evaluator


@hydra.main(version_base="v1.3", config_path="configs", config_name="full_sft")
def train(cfgs: OmegaConf):
    # Parallel accelerator initialization
    # Including mixed_precision, accum_gradient, tensorboard, etc.
    accelerator = Accelerator()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # SFT training, train_model contains trainable parameters.
    train_model: FastWAM = instantiate(cfgs.model)

    # Load from pretrained weights for initialization, the trained weights
    # for training resuming are handled in trainer.
    train_model.from_pretrained(
        cfgs.model.pretrained_model_name_or_path,
        device=accelerator.device,
        dtype=weight_dtype,
    )

    # Data module initialization
    data_module: DataModule = instantiate(cfgs.data_module)

    # Optional evaluator configuration
    evaluator: Evaluator = instantiate(cfgs.evaluator)

    # Configure trainer and start training
    trainer: BaseTrainer = instantiate(cfgs.trainer)

    trainer.train(
        accelerator=accelerator,
        model=train_model,
        train_loader=data_module.train_loader,
        eval_loader=data_module.eval_loader,
        evaluator=evaluator,
    )
    accelerator.end_training()
