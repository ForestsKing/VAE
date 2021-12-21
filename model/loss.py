import torch
from torch import nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, kld_weight=0.005):
        super(Loss, self).__init__()
        self.kld_weight = kld_weight

    def forward(self, recons, input, mu, log_var):
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + self.kld_weight * kld_loss
        return loss
