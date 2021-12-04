import wandb

with wandb.init(project="deneme", mode="online") as run:
    losses = [1, 2, 3, 4, 5]
    logdict = {
        "lossess": losses,
    }
    wandb.log(logdict)
