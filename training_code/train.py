import os
import pandas as pd
import numpy as np

from utils.options import args
import utils.common as utils

from importlib import import_module

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

from torch.optim.lr_scheduler import MultiStepLR as MultiStepLR
from torch.optim.lr_scheduler import StepLR as StepLR

from data import dataPreparer

import warnings, math

warnings.filterwarnings("ignore")

device = torch.device(f"cuda:{args.gpus[0]}")

checkpoint = utils.checkpoint(args)

def main(classnum):

    start_epoch = 0
    best_acc = 0.0
 
    # Data loading
    print('=> Preparing data..')
 
    # data loader
    loader = dataPreparer.Data(args, 
                               data_path=args.src_data_path, 
                               label_path=args.src_label_path,
                               classnum=classnum)
    data_loader = loader.loader_train
    data_loader_eval = loader.loader_test
    data_loader_test = loader.loader_final_test
    
    # Create model
    print('=> Building model...')

    # load training model
    # model = import_module(f'model.{args.arch}').__dict__[args.model]().to(device)
    model = models.mobilenet_v2()
    model.classifier[1] = nn.Linear(model.last_channel, classnum)
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
    model.to(device)

    # Load pretrained weights
    if args.pretrained:
 
        ckpt = torch.load(args.source_dir + args.source_file, map_location = device)
        state_dict = ckpt['state_dict']

        model.load_state_dict(state_dict)
        model = model.to(device)
    
    if args.test_only:
        # print(model)
        acc = inference(args, data_loader_test, model, args.output_file)
        print(f'Test acc {acc:.3f}\n')
        return
        
    if args.inference_only:
        # print(model)
        acc = inference(args, data_loader_eval, model, args.output_file)
        print(f'Test acc {acc:.3f}\n')
        return

    param = [param for name, param in model.named_parameters()]
    
    optimizer = optim.Adam(param, lr = args.lr, betas = (0.9, 0.999))
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma = args.lr_gamma)

    if args.resume:
        ckpt = torch.load(args.resume, map_location = device)
        state_dict = ckpt['state_dict']

        model.load_state_dict(state_dict)
        model = model.to(device)
    
        # ckpt = torch.load(args.source_dir + args.source_file, map_location = device)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])


    csv_data = [[], [], [], [], []]
    plot_file_name = args.output_file.replace('output.csv', 'learning_curve.csv')
    for epoch in range(start_epoch, args.num_epochs):
        saveCSV(csv_data, plot_file_name)
        scheduler.step(epoch)
        
        train(args, data_loader, model, optimizer, epoch, csv_data)
        
        (test_acc, test_loss) = test(args, data_loader_eval, model)
        csv_data[0].append(epoch + 1)
        csv_data[1].append(np.nan)
        csv_data[2].append(np.nan)
        csv_data[3].append(float(test_acc))
        csv_data[4].append(float(test_loss))
   
        is_best = best_acc < test_acc
        best_acc = max(best_acc, test_acc)
        

        state = {
            'state_dict': model.state_dict(),
            
            'optimizer': optimizer.state_dict(),
            
            'scheduler': scheduler.state_dict(),
            
            'epoch': epoch + 1
        }
        checkpoint.save_model(state, epoch + 1, is_best)
        
    inference(args, data_loader_eval, model, args.output_file)
    
    print(f'Best acc: {best_acc:.3f}\n')

def saveCSV(csv_data, output_file_name):
    output_file = dict()
    output_file['epoch'] = csv_data[0]
    output_file['train accuracy'] = csv_data[1]
    output_file['train loss'] = csv_data[2]
    output_file['test accuracy'] = csv_data[3]
    output_file['test loss'] = csv_data[4]
    output_file = pd.DataFrame.from_dict(output_file)
    output_file.to_csv(output_file_name, index = False)
    return output_file
  
       
def train(args, data_loader, model, optimizer, epoch, csv_data):
    losses = utils.AverageMeter()

    acc = utils.AverageMeter()

    criterion = nn.CrossEntropyLoss()
    
    num_iterations = len(data_loader)
    
    # switch to train mode
    model.train()
        
    for i, (inputs, targets, _) in enumerate(data_loader, 1):
        
        num_iters = num_iterations * epoch + i

        inputs = inputs.to(device)
        targets = targets.to(device)

    
        optimizer.zero_grad()
        
        # train
        output = model(inputs)

        loss = criterion(output, targets)

        # optimize cnn
        loss.backward()
        optimizer.step()

        ## train weights        
        losses.update(loss.item(), inputs.size(0))
        
        ## evaluate
        prec1, _ = utils.accuracy(output, targets, topk = (1, 5))
        acc.update(prec1[0], inputs.size(0))

        
        if i % args.print_freq == 0:     
            print(
                'Epoch[{0}]({1}/{2}): \n'
                'Train_loss: {train_loss.val:.4f} ({train_loss.avg:.4f})\n'
                'Train acc {acc.val:.3f} ({acc.avg:.3f})\n'.format(
                epoch, i, num_iterations, 
                train_loss = losses,
                acc = acc))
            csv_data[0].append(epoch + i / num_iterations)
            csv_data[1].append(float(acc.avg))
            csv_data[2].append(float(losses.avg))
            csv_data[3].append(np.nan)
            csv_data[4].append(np.nan)
            plot_file_name = args.output_file.replace('output.csv', 'learning_curve.csv')
            saveCSV(csv_data, plot_file_name)
      
 
def test(args, loader_test, model):
    losses = utils.AverageMeter()
    acc = utils.AverageMeter()

    criterion = nn.CrossEntropyLoss()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets, datafile) in enumerate(loader_test, 1):
            
            inputs = inputs.to(device)
            targets = targets.to(device)
             
            preds = model(inputs)
            loss = criterion(preds, targets)
        
            # image classification results
            prec1, _ = utils.accuracy(preds, targets, topk = (1, 5))
            losses.update(loss.item(), inputs.size(0))
            acc.update(prec1[0], inputs.size(0))

 
    print(f'Test acc {acc.avg:.3f}\n')

    return (acc.avg, losses.avg)
    

def inference(args, loader_test, model, output_file_name):
    losses = utils.AverageMeter()
    acc = utils.AverageMeter()

    criterion = nn.CrossEntropyLoss()
    outputs = []
    datafiles = []
    count = 1
    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets, datafile) in enumerate(loader_test, 1):
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
          
            preds = model(inputs)
    
            loss = criterion(preds, targets)
        
            # image classification results
            prec1, _ = utils.accuracy(preds, targets, topk = (1, 5))
            losses.update(loss.item(), inputs.size(0))
            acc.update(prec1[0], inputs.size(0))
            
            _, output = preds.topk(1, 1, True, True)
            
            outputs.extend(list(output.reshape(-1).cpu().detach().numpy()))
            
            datafiles.extend(list(datafile))
            
            count += inputs.size(0)
    

    output_file = dict()
    output_file['image_name'] = datafiles
    output_file['label'] = outputs
    
    output_file = pd.DataFrame.from_dict(output_file)
    output_file.to_csv(output_file_name, index = False)
    
    return acc.avg
  

if __name__ == '__main__':
    main(20)

