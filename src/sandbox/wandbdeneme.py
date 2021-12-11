import wandb, torch, timm

with wandb.init(project="deneme", mode="online") as run:
    model = timm.create_model("resnet18")

    torch.save(model, f="mymodel.pt")
    wandb.log_artifact("mymodel.pt", name="new_artifact", type="my_dataset")
