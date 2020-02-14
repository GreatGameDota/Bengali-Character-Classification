import torch
import torch.nn as nn
import torch.nn.functional as F

def criterion1(pred1, pred2, pred3, targets):
  l1 = F.cross_entropy(pred1, targets[:,0])
  l2 = F.cross_entropy(pred2, targets[:,1])
  l3 = F.cross_entropy(pred3, targets[:,2])
  return l1, l2, l3

def get_criterion():
    return criterion1