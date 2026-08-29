import hydra
from omegaconf import OmegaConf



@hydra.main(version_base="v1.3", config_path="configs", config_name="full_sft")
def train(cfgs: OmegaConf):

    pass