import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Mish

from pytorchcv.model_provider import get_model

num_classes_1 = 168
num_classes_2 = 11
num_classes_3 = 7

class Pooling(nn.Module):
  def __init__(self):
    super(Pooling, self).__init__()
    
    self.p1 = nn.AdaptiveAvgPool2d((1,1))
    self.p2 = nn.AdaptiveMaxPool2d((1,1))

  def forward(self, x):
    x1 = self.p1(x)
    x2 = self.p2(x)
    return (x1+x2) * 0.5

class Head(torch.nn.Module):
  def __init__(self, in_f, out_f):
    super(Head, self).__init__()
    
    self.f = nn.Flatten()
    self.l = nn.Linear(in_f, 512)
    self.m = Mish()
    # self.g1 = nn.GroupNorm(32, in_f)
    # self.g2 = nn.GroupNorm(32, 512)
    self.d = nn.Dropout(0.5)
    self.o = nn.Linear(512, out_f)
    # self.o = nn.Linear(in_f, out_f)
    self.b1 = nn.BatchNorm1d(in_f)
    self.b2 = nn.BatchNorm1d(512)

  def forward(self, x):
    # x = self.g1(x)
    x = self.f(x)
    x = self.b1(x)
    # x = self.g1(x)
    x = self.d(x)

    x = self.l(x)
    x = self.m(x)
    x = self.b2(x)
    # x = self.g2(x)
    x = self.d(x)

    out = self.o(x)
    return out

class FCN(torch.nn.Module):
  def __init__(self, base, in_f):
    super(FCN, self).__init__()
    self.base = base
    self.h1 = Head(in_f, num_classes_1)
    self.h2 = Head(in_f, num_classes_2)
    self.h3 = Head(in_f, num_classes_3)
  
  def forward(self, x):
    x = self.base(x)
    return self.h1(x), self.h2(x), self.h3(x)

def load_model(name, checkpoint_path=None, pretrained=False):
    model = get_model(name, pretrained=pretrained)
    model = nn.Sequential(*list(model.children())[:-1]) # Remove original output layer
    
    # model[0].final_pool = nn.Sequential(nn.BatchNorm2d(2048), Pooling(), Mish())
    model[0].final_pool = nn.Sequential(Pooling())

    model = FCN(model, 2048)

    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))
    
    return model
