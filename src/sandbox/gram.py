import torch, timm


class ConvActivationsWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.conv_outputs = []
        self.submodule_names = [name for name, layer in self.model.named_modules()]

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
        outputs, activations = self.model(x), self.conv_outputs
        return outputs, activations


model = ConvActivationsWrapper(timm.create_model("resnet18"))
input1 = torch.randn(2, 3, 224, 224)
input2 = torch.randn(2, 3, 224, 224)
output1, activations1 = model(input1)
output2, activations2 = model(input2)

stat1 = activations1[8]["output"]
stat2 = activations2[8]["output"]

gram = torch.einsum("bnhw,bnhw->bn", stat1, stat2)
