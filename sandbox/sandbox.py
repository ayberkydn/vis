import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import os


@hydra.main(config_path="configs", config_name="config")
def my_app(cfg: DictConfig) -> None:

    print(hydra.utils.get_original_cwd())
    print(os.getcwd())


if __name__ == "__main__":
    my_app()
