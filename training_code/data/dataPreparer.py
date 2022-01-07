from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import numpy as np
import pandas as pd
from torchvision.io import read_image
import os
from utils.options import args
from PIL import Image, ImageFilter

class DataPreparation(Dataset):
    def __init__(self, root=args, data_path=None, label_path=None,
                 transform=None, target_transform=None, classnum=20):
        
        self.root = root
        self.data_path = data_path 
        self.label_path = label_path 
        
        self.transform = transform
        self.target_transform = target_transform
        
        ## preprocess files
        self.preprocess(self.data_path, self.label_path, classnum)
        

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_idx = idx # % len(self.data_files)
        data_file = self.data_files[file_idx]
        img_path = os.path.join(self.data_path, data_file)
        image = Image.open(img_path) # plt.imread(img_path)
    
        # label = int(np.random.rand(1)[0] * 20)
        # image = image.filter(ImageFilter.GaussianBlur(radius = label))
        if self.transform:
            image = self.transform(image)
            
        if self.label_path is None:
            return image, -1, data_file
        
        label = self.file_labels['label'][self.file_labels['file'] == data_file].iloc[0]

        if self.target_transform:
            label = self.target_transform(label)


        return image, label, data_file
    
    def preprocess(self, data_path, label_path, classnum):
        self.data_files = os.listdir(data_path)
        self.data_files.sort()

        if label_path is not None:
            self.file_labels = pd.read_csv(label_path)
            self.file_labels = self.file_labels.loc[self.file_labels['label'] < classnum].reset_index(drop=True)
            self.data_files = self.file_labels['file'].tolist()


class Data:
    def __init__(self, args, data_path, label_path, classnum):
        

        # transform_train = transforms.Compose([
        #     # transforms.RandomResizedCrop((28, 28), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        #     # transforms.RandomCrop(31),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=(0.5), std=(0.5)),
        #     # transforms.RandomRotation(45)
        # ])

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485), std=(0.229)),
        ])
        
        train_dataset = DataPreparation(root=args,  
                                        data_path=data_path,
                                        label_path=label_path,
                                        transform=transform,
                                        classnum=classnum)
        
        self.loader_train = DataLoader(
            train_dataset, batch_size=args.train_batch_size, shuffle=True, 
            num_workers=2
            )
        
        
        test_data_path = data_path.replace('train', 'valid')
        test_label_path = label_path
        final_test_label_path = label_path
        final_test_data_path = data_path.replace('train', 'test')

        if label_path is not None:
            test_label_path = label_path.replace('train', 'valid')
            final_test_label_path = label_path.replace('train', 'test')
        
        test_dataset = DataPreparation(root=args,  
                                       data_path=test_data_path,
                                       label_path=test_label_path,
                                       transform=transform,
                                       classnum=classnum)
        
        self.loader_test = DataLoader(
            test_dataset, batch_size=args.train_batch_size, shuffle=False, 
            num_workers=2
            )

        final_test_dataset = DataPreparation(root=args,  
                                             data_path=final_test_data_path,
                                             label_path=final_test_label_path,
                                             transform=transform,
                                             classnum=classnum)
        self.loader_final_test = DataLoader(
            final_test_dataset, batch_size=args.final_test_batch_size, shuffle=False, 
            num_workers=2
            )
       
       
