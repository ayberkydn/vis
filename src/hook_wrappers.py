import timm
import torch


class AuxLossWrapper(torch.nn.Module):
    def __init__(self, model, n=0, mode="all"):
        super().__init__()
        self.model = model
        self.losses = []

        # def conv_hook(module, inputs, outputs):
        #     out_std = outputs.std(dim=0)
        #     self.losses.append(-out_std.mean())

        def bn_hook(module, inputs, outputs):
            assert len(inputs) == 1
            inputs = inputs[0]

            inputs_mean = inputs.mean(dim=[0, -1, -2])
            inputs_var = inputs.var(dim=[0, -1, -2])
            running_mean = module.running_mean
            running_var = module.running_var

            mean_loss = torch.square(inputs_mean - running_mean)
            var_loss = torch.square(inputs_var - running_var)
            # loss = torch.mean(mean_loss + var_loss)
            loss = torch.mean(mean_loss)

            self.losses.append(loss)

        layers = [layer for name, layer in self.model.named_modules()]
        bn_layers = [l for l in layers if isinstance(l, torch.nn.BatchNorm2d)]
        if mode == "all":
            loss_layers = bn_layers
        elif mode == "first":
            assert n > 0
            loss_layers = bn_layers[:n]
        elif mode == "last":
            assert n > 0
            loss_layers = bn_layers[-n:]
        elif mode == "particular":
            assert n < len(bn_layers)
            loss_layers = [bn_layers[n]]
        else:
            raise Exception("invalid mode for auxloss")

        for layer in loss_layers:
            layer.register_forward_hook(bn_hook)

    def forward(self, x):
        del self.losses
        self.losses = []
        out = self.model(x)
        loss = torch.mean(torch.stack(self.losses))
        return out, loss


if __name__ == "__main__":
    model = timm.create_model("resnet18")
