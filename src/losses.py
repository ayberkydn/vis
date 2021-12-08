import kornia
import torch


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(
        pred, y_b
    )


# def diversity_loss(conv_activations):
#     losses = []
#     for act in activations:
#         outputs = act["output"]
#         bsize = outputs.shape[0] // 2
#         act1 = outputs[:bsize]
#         act2 = outputs[bsize:]
#         loss = -torch.mean(torch.abs(act1 - act2))
#         losses.append(loss)
#     return torch.mean(torch.stack(losses))


def diversity_loss(activations):
    losses = []
    for act_n in range(len(activations)):
        var = activations[act_n]["inputs_var"]
        loss = -torch.sqrt(var)
        loss.append(loss.mean())

    return torch.mean(torch.stack(losses))


def mean_tv_loss(imgs):
    C, H, W = imgs.shape[-3], imgs.shape[-2], imgs.shape[-1]
    return kornia.losses.total_variation(imgs).mean() / (C * H * W)


def score_increase_loss(logits, classes):
    return -logits[torch.arange(len(logits)), classes].mean()


def softmax_loss(logits, classes, T=1, smooth=0):
    return torch.nn.functional.cross_entropy(
        input=logits / T,
        target=classes,
        label_smoothing=smooth,
    )


def kl_loss_gaussian(mean1, var1, mean2, var2, eps=1e-7):
    var1 = var1 + eps
    loss = torch.log(torch.sqrt(var1) / torch.sqrt(var2)) - 0.5 * (
        1 - ((var2 + torch.square(mean1 - mean2)) / var1)
    )
    return loss


def bn_stats_loss(activations):
    losses = []
    for act_n in range(len(activations)):
        act_mean = activations[act_n]["inputs_mean"]
        act_var = activations[act_n]["inputs_var"]
        running_mean = activations[act_n]["module"].running_mean
        running_var = activations[act_n]["module"].running_var

        # loss = kl_loss_gaussian(act_mean, act_var, running_mean, running_var)
        loss = (act_mean - running_mean) ** 2 + (act_var - running_var) ** 2

        losses.append(loss.mean())

    return torch.mean(torch.stack(losses))
