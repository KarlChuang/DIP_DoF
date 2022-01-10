import os
import pandas as pd
import numpy as np

from training_code.utils.options import args
import training_code.utils.common as utils

from importlib import import_module

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from torch.optim.lr_scheduler import MultiStepLR as MultiStepLR
from torch.optim.lr_scheduler import StepLR as StepLR
from PIL import Image, ImageFilter
from torchvision.io import read_image

import warnings, math

warnings.filterwarnings("ignore")

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

checkpoint = utils.checkpoint(args)


class DataPreparation(Dataset):
    def __init__(self,
                 root=args,
                 img_path=None,
                 step=8,
                 transform=None):

        self.root = root
        self.img_path = img_path
        self.step = step
        self.transform = transform
        ## preprocess files
        if not (self.img_path is None):
            self.preprocess(self.img_path, self.step)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        image = self.data_files[idx]
        (h, w) = self.data_indexs[idx]
        if self.transform:
            image = self.transform(image)
        return image, -1, h, w

    def preprocess(self, img_path, step):
        img = Image.open(img_path).convert('L')
        # img = img.filter(ImageFilter.GaussianBlur(radius=5))
        # img.show()
        (width, height) = img.size
        # img = img.resize((int(width / 2), int(height / 2)))
        # (width, height) = img.size
        top = 0
        self.data_files = []
        self.data_indexs = []
        while top + 32 <= height:
            left = 0
            while left + 32 <= width:
                imgCrop = img.crop((left, top, left + 32, top + 32))
                self.data_files.append(imgCrop)
                self.data_indexs.append((top, left))
                left += step
            top += step



def main(img_path, model_path, out_path, step, test_batch_size, classnum):
    # Data loading
    print('=> Preparing data..')

    # data loader
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485), std=(0.229)),
    ])
    test_dataset = DataPreparation(args,
                                  img_path=img_path,
                                  step=step,
                                  transform=transform)
    loader_test = DataLoader(test_dataset,
                             batch_size=test_batch_size,
                             shuffle=True,
                             num_workers=2)

    # Create model
    print('=> Building model...')

    # load training model
    # model = import_module(f'model.{args.arch}').__dict__[args.model]().to(device)
    model = models.mobilenet_v2()
    model.classifier[1] = nn.Linear(model.last_channel, classnum)
    model.features[0][0] = nn.Conv2d(1,
                                     32,
                                     kernel_size=3,
                                     stride=2,
                                     padding=1,
                                     bias=False)

    model.to(device)

    # Load pretrained weights
    ckpt = torch.load(model_path,
                        map_location=device)
    state_dict = ckpt['state_dict']

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    classes = np.array([i for i in range(classnum)])


    img = Image.open(img_path).convert('L')
    # img = img.resize((int(img.size[0] / 2), int(img.size[1] / 2)))

    imgArr = np.array(img)

    depth = np.zeros(imgArr.shape)
    base = np.zeros(imgArr.shape)
    num_itr = len(loader_test)
    with torch.no_grad():
        for i, (inputs, targets, h, w) in enumerate(loader_test, 1):
            print('{0}/{1}'.format(i, num_itr))
            inputs = inputs.to(device)
            preds = model(inputs)
            maxk = max((1,))
            _, pred = preds.topk(maxk, 1, True, True)
            pred = pred.t()
            for idx, d in enumerate(pred[0]):
                depth[h[idx]:h[idx]+32,w[idx]:w[idx]+32] += float(d)
                base[h[idx]:h[idx]+32,w[idx]:w[idx]+32] += 1
    base[base == 0] = 1
    depth = (depth / base / (classnum - 1) * 255).astype('uint8')
    data = Image.fromarray(depth, 'L')
    data.save(out_path)
    # data.show()
    # print(depthMap)

if __name__ == '__main__':
    main('test.png', './result/mobile5-2/checkpoint/model_best.pt', 'test_depth.png', 2, 128, 20)
