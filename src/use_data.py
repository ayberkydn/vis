import torch, wandb

best_model = wandb.restore("in_layer.pt", run_path="ayberkydn/vis-denemeler/380mr8sv")
model = torch.load(best_model.name)
