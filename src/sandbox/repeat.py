import torch


a = torch.randn(50, requires_grad=True)
b = torch.randn(50, requires_grad=True)
c = a + b

loss = c.mean()
a_grad = torch.autograd.grad(loss, a, create_graph=True)[0]
mean_a_grad = a_grad.mean()
