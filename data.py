import os
import torch
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ImageDataSet(Dataset):
    def __init__(self, path, nums, train=False, transform=None):
        self.labels = pd.read_csv('./data/trainLabels.csv')
        self.classes = {}
        self.transform = transform
        self.imgs = []
        self.train = train

        with open('./class.json', 'r') as f:
            self.classes = json.load(f)

        #paths = os.listdir(path)
        self.imgs = [os.path.join(path, str(i) + '.png') for i in range(1, nums + 1)] 
        

    def __getitem__(self, item):
        img = Image.open(self.imgs[item])
        if self.transform != None:
            img = self.transform(img)
        if self.train:
            label = self.classes[self.labels.label[item]]
            return img, label
        else:
            return img

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    trainpath = './data/train'
    trainset = ImageDataSet(trainpath, 50000, transform=transforms.ToTensor())
    first = next(iter(trainset))
    print(first)

    

    
        