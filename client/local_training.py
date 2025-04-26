# client/local_training.py
import torch, torchvision, torchvision.transforms as T

def load_data(_cfg):
    # CIFAR10 → MNIST example
    ds_train = torchvision.datasets.MNIST(
        "./data", train=True, download=True,
        transform=T.ToTensor()
    )
    ds_val = torchvision.datasets.MNIST(
        "./data", train=False, download=True,
        transform=T.ToTensor()
    )
    return (
        torch.utils.data.DataLoader(ds_train, batch_size=32, shuffle=True),
        torch.utils.data.DataLoader(ds_val, batch_size=32),
    )

import numpy as np
from collections import OrderedDict

def get_parameters(net):
    return [val.cpu().detach().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters):
    state_dict = OrderedDict()
    for (k, _), v in zip(net.state_dict().items(), parameters):
        state_dict[k] = torch.tensor(v)
    net.load_state_dict(state_dict, strict=True)

def train(net, loader, epochs=1):
    net.train()
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            pred = net(xb)
            loss_fn(pred, yb).backward()
            opt.step()

def test(net, loader):
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            out = net(xb).argmax(dim=1)
            correct += (out == yb).sum().item()
            total += yb.size(0)
    return 1 - correct/total, correct/total
