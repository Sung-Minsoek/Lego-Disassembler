import numpy as np
import pandas as pd
import random

import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None, no_labels=False):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.no_labels = no_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.no_labels:
            image = self.data[idx]
            if self.transform:
                image = self.transform(image)

            return image
        else:
            image = self.data[idx]
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label


def data_and_labels_to_tensor(data_path, labels_path):
    data_np = np.load(data_path)
    data_tensor = torch.from_numpy(data_np)
    data_tensor[data_tensor == 45] = 0        # background to 0

    labels_csv = pd.read_csv(labels_path)
    labels_np = np.array(labels_csv.values[:,1:])
    labels_np = labels_np.astype(np.float32)
    labels_tensor = torch.from_numpy(labels_np)

    # one-hot labels to index labels
    index_list = []

    for row in labels_tensor:
        indices = torch.nonzero(row).squeeze().tolist()
        index_list.append(indices)

    labels_tensor = torch.tensor(index_list)
    
    return data_tensor, labels_tensor


def make_dataloader(data, labels, params, random_seed=0):
    train_data, valid_data, train_labels, valid_labels = train_test_split(data, labels, test_size=params['valid_size'], random_state=random_seed)

    train_transforms = v2.Compose([
        v2.Resize((224, 224), antialias=False),
        v2.RandomHorizontalFlip(),
        v2.RandomCrop((224, 224), padding=16),
        v2.ToDtype(torch.float32),
        v2.Normalize((0.45,), (0.225,))
    ])

    valid_transforms = v2.Compose([
        v2.ToDtype(torch.float32),
        v2.Normalize((0.45,), (0.225,))
    ])

    train_dataset = CustomDataset(train_data, train_labels, transform=train_transforms)
    trainloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=params['worker'])

    valid_dataset = CustomDataset(valid_data, valid_labels, transform=valid_transforms)
    validloader = DataLoader(valid_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=params['worker'])
    
    return trainloader, validloader
