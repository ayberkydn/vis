import timm
import torch


class IntermediateOutput:
    def __init__(self, module, name, inputs, outputs):
        self.name = name
        self.module = module
        self.inputs = inputs
        self.outputs = outputs


class ConvActivationsWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.conv_outputs = []

        def create_hook(name):
            def hook(module, inputs, outputs):
                assert len(inputs) == 1
                self.conv_outputs.append(
                    {
                        "name": name,
                        "module": module,
                        "output": outputs,
                    }
                )

            return hook

        for name, layer in self.model.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                layer.register_forward_hook(create_hook(name))

    def forward(self, x):
        self.conv_outputs = []
        outputs, activations = self.model(x), self.conv_outputs
        return outputs, activations


class BNStatsWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.bn_stats = []

        def create_hook(name):
            def hook(module, inputs, outputs):
                assert len(inputs) == 1
                self.bn_stats.append(
                    {
                        "name": name,
                        "module": module,
                        "inputs_mean": inputs[0].mean(dim=[0, -1, -2]),
                        "inputs_var": inputs[0].var(dim=[0, -1, -2]),
                    }
                )

            return hook

        for name, layer in self.model.named_modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.register_forward_hook(create_hook(name))

    def forward(self, x):
        self.bn_stats = []
        outputs, activations = self.model(x), self.bn_stats
        return outputs, activations


class VerboseExecutionWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.aux = []

        def create_hook(name):
            def hook(module, inputs, outputs):
                assert len(inputs) == 1
                self.aux.append(
                    {
                        "name": name,
                        "module": module,
                        "inputs": inputs,
                        "output": outputs,
                    }
                )

            return hook

        for name, layer in self.model.named_modules():
            layer.register_forward_hook(create_hook(name))

    def forward(self, x):
        self.aux = []
        outputs, aux = self.model(x), self.aux
        return outputs, aux


class ConvSimilarityLossWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.losses = []

    def forward(self, x):
        del self.losses
        self.losses = []
        outputs = self.model(x)
        loss = torch.mean(torch.stack(self.losses))
        return outputs, loss


class AuxLossWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.losses = []

        def conv_hook(module, inputs, outputs):
            out_std = outputs.std(dim=0)
            self.losses.append(-out_std.mean())

        def bn_hook(module, inputs, outputs):
            assert len(inputs) == 1
            inputs = inputs[0]

            inputs_mean = inputs.mean(dim=[0, -1, -2])
            inputs_var = inputs.var(dim=[0, -1, -2])
            running_mean = module.running_mean
            running_var = module.running_var

            mean_loss = torch.square(inputs_mean - running_mean)
            var_loss = torch.square(inputs_var - running_var)
            loss = torch.mean(mean_loss + var_loss)

            self.losses.append(loss)

        for name, layer in self.model.named_modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.register_forward_hook(bn_hook)
            # if isinstance(layer, torch.nn.Conv2d):
            #     layer.register_forward_hook(conv_hook)

    def forward(self, x):
        del self.losses
        self.losses = []
        out = self.model(x)
        loss = torch.mean(torch.stack(self.losses))
        return out, loss


if __name__ == "__main__":
    model = timm.create_model("resnet18")
    model = BNStatsLossWrapper(model)
    inp = torch.randn(7, 3, 224, 224)
    logits, aux = model(inp)
