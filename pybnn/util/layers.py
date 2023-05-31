import torch
import torch.nn as nn
import numpy as np


class AppendLayer(nn.Module):
    def __init__(self, noise=1e-3, device="cpu", *args, **kwargs):
        factory_kwargs = {'device': device}
        super().__init__(*args, **kwargs)

        self.log_var = nn.Parameter(torch.ones(1, 1, device=device))

        nn.init.constant_(self.log_var, val=np.log(noise))

    def forward(self, x):
        return torch.cat((x, self.log_var.to(x) * torch.ones_like(x).to(x)), dim=-1)
