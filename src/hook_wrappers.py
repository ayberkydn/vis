import torch


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


class BNStatsModelWrapper(torch.nn.Module):
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
