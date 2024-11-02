import torch
from torch import nn


class CCCLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gold, pred, weights=None):

        gold_mean = torch.mean(gold, 1, keepdim=True, out=None)
        pred_mean = torch.mean(pred, 1, keepdim=True, out=None)
        covariance = (gold - gold_mean) * (pred - pred_mean)
        gold_var = torch.var(gold, 1, keepdim=True, unbiased=True, out=None)
        pred_var = torch.var(pred, 1, keepdim=True, unbiased=True, out=None)
        ccc = 2. * covariance / (
                (gold_var + pred_var + torch.mul(gold_mean - pred_mean, gold_mean - pred_mean)) + 1e-50)
        ccc_loss = 1. - ccc

        if weights is not None:
            ccc_loss *= weights

        return torch.mean(ccc_loss)
