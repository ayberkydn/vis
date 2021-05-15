import hydra
from src.train import train
from omegaconf import DictConfig, OmegaConf
import os

os.environ["HYDRA_FULL_ERROR"] = "0"


@hydra.main(config_path="configs", config_name="defaults.yaml")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    train(cfg)


if __name__ == "__main__":
    main()
