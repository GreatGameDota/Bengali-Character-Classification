import torch
from torchtoolbox.optimizer import Lookahead

def get_optimizer(model, lr=0.001, lookahead=False):
    base_opt = torch.optim.Adam(model.parameters(), lr=lr)
    if lookahead:
        optimizer = Lookahead(base_opt)
        optimizer.defaults = base_opt.defaults
        return optimizer
    return base_opt