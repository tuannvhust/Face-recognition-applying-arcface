from turtle import forward
import torch.nn as nn
import torch

class FocalLoss(nn.Module):
    def __init__(self,gamma=0,eps=1e-7):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()
    def forward(self,input,target):
        logp = self.ce(input,target)
        p = torch.exp(-logp)
        loss = (1-p)**self.gamma*logp
        return loss 
        


