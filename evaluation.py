# Evaluation entrypoint
# - hydra-driven configurations

import hydra
from omegaconf import OmegaConf


@hydra.main(version_base="v1.3", config_path="configs", config_name="evaluation")
def evaluation(cfgs: OmegaConf):

    pass
