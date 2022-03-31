import argparse
from ast import parse
from turtle import window_height
import torch

def get_args():
    parser = argparse.ArgumentParser(description='Face Recognition Trainer')

    parser.add_argument('--scenario',type=str,default='train')
    parser.add_argument('--input_shape',type=str,default='1,128,128')
    parser.add_argument('--train_root',type=str,default='data/Datasets/webface/CASIA-maxpy-clean')
    parser.add_argument('--train_batch_size',type=int,default=16)
    parser.add_argument('--num_workers',type=int,default=4)
    parser.add_argument('--test_list_path',type=str,default='data/Datasets/lfw/lfw_test_pair.txt')
    parser.add_argument('--test_root',type=str,default='data/Datasets/lfw/lfw-align-128')
    parser.add_argument('--model',type=str,default='resnet_face18')
    parser.add_argument('--use_se',action='store_false')
    parser.add_argument('--loss',type=str,default='focal_loss')
    parser.add_argument('--in_features',type=int,default=512)
    parser.add_argument('--out_features',type=int,default=13938)
    parser.add_argument('--s',type=int,default=30)
    parser.add_argument('--learning_rate',type=float,default=1e-1)
    parser.add_argument('--weight_decay',type=float,default=5e-4)
    parser.add_argument('--learning_step',type=int,default=10)
    parser.add_argument('--num_epoch',type=int,default=50)
    parser.add_argument('--print_freq',type=int,default=100)
    parser.add_argument('--save_interval',type=int,default=10)
    parser.add_argument('--max_epoch',type=int,default=50)
    #parser.add_argument('--checkpoints_path',type=str,default='checkpoints')
    parser.add_argument('--metric',type=str,default='arcface')
    parser.add_argument('--optimizer',type=str,default='sgd')


    











    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args