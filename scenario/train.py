from ast import arg
from random import shuffle
import torch
from torchvision import transforms as T
import torchvision
from torch.nn import DataParallel
from torch.utils import data
import utils
import os
import nnet
from torch.optim.lr_scheduler import StepLR
import time
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
from .test import*


def train(args):
    #device = torch.device('cuda')
    log_folder = utils.get_log_folder(args)
    writer = SummaryWriter(log_folder)
    input_shape_list = [int(_) for _ in args.input_shape.split(',') ]
    train_transform = T.Compose([T.Grayscale(),T.RandomCrop(input_shape_list[1:]),
                                    T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize(mean=[0.5],std=[0.5])])

    train_dataset = torchvision.datasets.ImageFolder(args.train_root,transform = train_transform)
    train_loader = data.DataLoader(train_dataset,batch_size=args.train_batch_size,shuffle= True,num_workers = args.num_workers)
    indentities_list = utils.get_test_list(args.test_list_path)
    image_path = [os.path.join(args.test_root,file) for file in indentities_list]
    print(f'{len(train_loader)} train iterations per epoch ')

    model = nnet.get_model(args).to(args.device)
    criterion = nnet.get_criterion(args)
    metric = nnet.get_metric(args)
    model = DataParallel(model)
    metric = DataParallel(metric)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params':model.parameters()},{'params':metric.parameters()}],lr= args.learning_rate,weight_decay=args.weight_decay)
    elif arg.optimizer =='adam':
        optimizer = torch.optim.Adam([{'params':model.parameters()},{'params':metric.parameters()}],lr= args.learning_rate,weight_decay=args.weight_decay)
    
    scheduler = StepLR(optimizer,step_size = args.learning_step,gamma=0.1)
    start = time.time()
    for i in range(args.num_epoch):
        scheduler.step()
        model.train()
        for batch_index,batch in enumerate(train_loader):
            data_input,label = batch
            data_input = data_input.to(args.device)
            label = label.to(args.device).long()
            feature = model(data_input)
            output = metric(feature,label)
            loss = criterion(output,label)
            optimizer.zero_grad()
            loss.backward()
            scheduler.step()
            
            iters = i*len(train_loader) + batch_index
            if iters % args.print_freq == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output,axis=1)
                label = label.data.cpu().numpy()
                accuracy = np.mean((output==label).astype(int))
                speed = args.print_freq/(time.time()-start)
                now_time = time.asctime(time.localtime(time.time()))
                print(f'{now_time},train epoch : {i}, iteration : {batch_index}, speed : {speed} iters/s, loss: {loss}, accuracy: {accuracy}')





                loss_metrics = {'loss':float(loss)}
                writer.add_scalars('loss',loss_metrics,iters)
                print(f"Evaluate epoch {i}_train_accuracy: {float(accuracy)}")
                with open(os.path.join(log_folder,'train_log.txt'),'a') as fin:
                    fin.write(f'Evaluate epoch {i} train_accuraacy: {float(accuracy)}')

                accuracy_metric = {'train_accuracy': float(accuracy)}
                writer.add_scalars('accuracy',accuracy_metric)
                start = time.time()

            
                
        if i % args.save_interval == 0 or i == args.max_epoch:
            checkpoints_path = utils.get_ckpt_folder(args)
            save_model(model,checkpoints_path, args.model, i)

        model.eval()
        test_list_path = args.test_list_path
        acc = lfw_test(model, image_path,indentities_list,args.test_list_path,args.test_batch_size)
        metric = {'test_accuracy':float(acc)}
        writer.add_scalars('test_accuracy',metric)






def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name

    
