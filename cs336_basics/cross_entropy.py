import torch


def cross_entropy(predicted: torch.Tensor, targets: torch.Tensor):
    pred_max_rows = torch.max(predicted, dim=-1, keepdim=True).values
    predicted = predicted - pred_max_rows

    denom = torch.log(torch.sum(torch.exp(predicted), dim=-1, keepdim=True))
    predicted = predicted - denom
    selected = predicted.gather(dim=-1, index=targets.unsqueeze(-1))
    return -selected.mean()
