import torch, wandb

with wandb.init(project="vis-denemeler", mode="offline") as run:
    artifact = run.use_artifact("my_inlayer_artifact:latest")
    artifact_dir = artifact.download()
