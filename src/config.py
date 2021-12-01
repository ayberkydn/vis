from dataclasses import dataclass


@dataclass
class TrainConfig:
    GPU: int
    IMG_SIZE: int
    NET_INPUT_SIZE: int
    NETWORKS: str
    LEARNING_RATE: float
    ITERATIONS: int
    BATCH_SIZE: int
    CLASS: int
    LOG_FREQUENCY: int
    PARAM_FN: str


cfg = TrainConfig(
    GPU=2,
    IMG_SIZE=512,
    NET_INPUT_SIZE=224,
    NETWORKS=[
        # "densenet121",
        "inception_v3",
        # "resnet50",
    ],
    LEARNING_RATE=0.01,
    ITERATIONS=1000000,
    BATCH_SIZE=32,
    CLASS=309,
    LOG_FREQUENCY=500,
    PARAM_FN="sigmoid",
)
