from torch import nn


def count_parameters(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "total_params": total_params,
        "in_billions": total_params / 1e9,
        "in_millions": total_params / 1e6,
    }


def count_trainable_parameters(model: nn.Module):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "trainable_params": trainable_params,
        "in_billions": trainable_params / 1e9,
        "in_millions": trainable_params / 1e6,
    }
