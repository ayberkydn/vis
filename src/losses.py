import kornia
import torch


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(
        pred, y_b
    )


def diversity_loss(activations):
    losses = []
    for act in activations:
        outputs = act["output"]
        bsize = outputs.shape[0] // 2
        act1 = outputs[:bsize]
        act2 = outputs[bsize:]
        loss = -torch.mean(torch.abs(act1 - act2))
        losses.append(loss)
    return torch.mean(torch.stack(losses))


def mean_tv_loss(imgs):
    H, W = imgs.shape[-2], imgs.shape[-1]
    return kornia.losses.total_variation(imgs).mean() / (H * W)


def score_maximizer_loss(logits, classes):
    # return -torch.gather(logits, -1, classes.unsqueeze(1), sparse_grad=True).mean()
    return -logits[torch.arange(len(logits)), classes].mean()


def probability_maximizer_loss(logits, classes):
    # b = logits.shape[0]
    # dev = logits.device

    # target = (
    #     torch.ones(size=[b], dtype=torch.long, device=logits.device) * target_classes
    # )
    return torch.nn.CrossEntropyLoss()(logits, classes)


def bn_stats_loss(activations):
    losses = []
    for act_n in range(len(activations)):
        act_mean = activations[act_n]["inputs_mean"]
        act_var = activations[act_n]["inputs_var"] + 1e-8
        running_mean = activations[act_n]["module"].running_mean
        running_var = activations[act_n]["module"].running_var

        loss_n = torch.log(torch.sqrt(act_var) / torch.sqrt(running_var)) - 0.5 * (
            1 - ((running_var + torch.square(act_mean - running_mean)) / act_var)
        )
        losses.append(loss_n.mean())

    loss = torch.mean(torch.stack(losses))
    return loss
