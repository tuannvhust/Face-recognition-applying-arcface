from ast import arg
from utils import args
from .loss import FocalLoss
from .models import*
from .metric import*

def get_metric(args):
    metric = {
        'arcface': ArcFace
    }

def get_model(args):
    models = {
        'resnet_face18': resnet_face18(use_se=args.use_se),
        'resnet_18': resnet18(),
        'resnet_34': resnet34(),
        'renet_101':resnet101(),
        'resnet_152':resnet152()

    }
    try:
        return models[args.model]
    except:
        raise NotImplementedError
def get_criterion(args):
    losses = {
        'focal_loss': FocalLoss(gamma=2),
        'ce': torch.nn.CrossEntropyLoss()
    }
    try: 
        return losses[args.loss]
    except:
        raise NotImplementedError
def get_metric(args):
    metrics = {
        'arcface': ArcFace(args).to(args.device),
        'cosface': CosFace(args).to(args.device),
        'sphereface':SphereFace(args).to(args.device)
    }
    try:
        return metrics[args.metric]
    except:
        raise NotImplementedError


