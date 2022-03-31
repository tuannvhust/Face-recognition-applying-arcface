
import torch.nn.functional as F
import torch.nn as nn
import torch
import math




class ArcFace(nn.Module):
    def __init__(self, args,m=0.35):
        super(ArcFace, self).__init__()
        #self.embedding = model
        self.s = args.s
        self.m = m
        self.W = nn.Parameter(torch.FloatTensor(args.out_features, args.in_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, y):
        # compute embedding
        #x = self.embedding(x)
        # normalize features
        x = F.normalize(x)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)

        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, y.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        return output


class CosFace(nn.Module):
    def __init__(self, args,m=0.5):
        super(CosFace, self).__init__()
        #self.embedding = model
        self.s = args.s
        self.m = m
        self.W = nn.Parameter(torch.FloatTensor(args.out_features, args.in_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, y):
        # compute embedding
        #x = self.embedding(x)
        # normalize features
        x = F.normalize(x)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)

        # add margin
        target_logits = logits - self.m
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, y.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        return output


class SphereFace(nn.Module):
    def __init__(self, args,m=4):
        super(SphereFace, self).__init__()
        #self.embedding = model
        self.s = args.s
        self.m = m
        self.W = nn.Parameter(torch.FloatTensor(args.out_features, args.in_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x, y):
        # compute embedding
        #x = self.embedding(x)
        # normalize features
        x = F.normalize(x)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)

        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(self.m * theta)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, y.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        return output