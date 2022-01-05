import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from utils.options import args
import utils.common as utils

from importlib import import_module

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR as MultiStepLR
from torch.optim.lr_scheduler import StepLR as StepLR

from data import dataPreparer

import warnings, math

warnings.filterwarnings("ignore")

device = torch.device(f"cuda:{args.gpus[0]}")

checkpoint = utils.checkpoint(args)

def main():
     # Data loading
    print('=> Preparing data..')

    # data loader
    loader = dataPreparer.Data(args,
                            data_path=args.src_data_path, 
                            label_path=args.src_label_path)
    
    data_loader = loader.loader_train
    data_loader_eval = loader.loader_test
    data_loader_test = loader.loader_final_test
    
    # Create model
    print('=> Building model...')
    model = import_module(f'model.{args.arch}').__dict__[args.model]().to(device)

    plot_file_name = args.plot_csv
    curve_pd = pd.read_csv(plot_file_name)
    plot_fig_name = plot_file_name.replace('.csv', '.png')
    plotting(curve_pd, plot_fig_name)

    ckpt = torch.load(args.source_dir + args.source_file, map_location = device)
    state_dict = ckpt['state_dict']
    model.load_state_dict(state_dict)
    model = model.to(device)

    plot_fig_name = plot_fig_name.replace('learning_curve', 'confussion_matrix')
    confussion(args, data_loader_eval, model, plot_fig_name)

def confussion(args, loader_test, model, plt_name):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for i, (inputs, targets, datafile) in enumerate(loader_test, 1):
            inputs = inputs.to(device)
            targets = targets.to(device)

            preds = model(inputs)
            _, preds = torch.max(preds, 1)
            y_pred.extend(preds.view(-1).detach().cpu().numpy())
            y_true.extend(targets.view(-1).detach().cpu().numpy())
    cf_matrix = confusion_matrix(y_true, y_pred)
    per_cls_acc = cf_matrix.diagonal() / cf_matrix.sum(axis=0)
    class_names = [str(i) for i in range(10)]

    df_cm = pd.DataFrame(cf_matrix, class_names, class_names)
    plt.figure(figsize = (9,6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
    plt.xlabel("prediction")
    plt.ylabel("label (ground truth)")
    plt.savefig(plt_name)

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

    return (acc.avg, losses.avg)

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

def plotting(pd_data, fig_name):
    fig, axs = plt.subplots(2, figsize=(12,8))
    fig.tight_layout(pad=3.0)
    axs[0].set_title('Learning curve -- CrossEntropyLoss')
    axs[0].plot(pd_data['epoch'], pd_data['train loss'], label='train loss')
    axs[0].plot(pd_data['epoch'], pd_data['test loss'], label='test loss')
    axs[0].set_xlabel("epoch")
    axs[0].legend(loc="upper left")
    axs[1].set_title('Learning curve -- Accuracy')
    axs[1].plot(pd_data['epoch'], pd_data['train accuracy'], label='train accuracy')
    axs[1].plot(pd_data['epoch'], pd_data['test accuracy'], label='test accuracy')
    axs[1].set_xlabel("epoch")
    axs[1].legend(loc="upper left")
    fig.savefig(fig_name)

if __name__ == '__main__':
    main()

