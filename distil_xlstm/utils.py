from torch import nn


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
