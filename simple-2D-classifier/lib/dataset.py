import numpy as np

import os

import torch
from torch.utils.data import Dataset
import math


class Simple2DDataset(Dataset):
    def __init__(self, split='train'):
        super().__init__()
        assert split in ['train', 'valid'], f'Split parameters "{split}" must be either "train" or "valid".'
        # Read either train or validation data from disk based on split parameter using np.load.
        # Data is located in the folder "data".
        # Hint: you can use os.path.join to obtain a path in a subfolder.
        # Save samples and annotations to class members self.samples and self.annotations respectively.
        # Samples should be an Nx2 numpy array. Annotations should be Nx1.

        path="data"
        if split == "train":
            data = np.load(os.path.join(path, "", "train.npz"))
        elif split == "valid":
            data =  np.load(os.path.join(path, "", "valid.npz"))
        self.samples = data['samples']
        self.annotations = data['annotations']
            
    def __len__(self):
        # Returns the number of samples in the dataset.
        return self.samples.shape[0]
    
    def __getitem__(self, idx):
        # Returns the sample and annotation with index idx.
        sample = self.samples[idx]
        annotation = self.annotations[idx]
        
        # Convert to tensor.
        return {
            'input': torch.from_numpy(sample).float(),
            'annotation': torch.from_numpy(annotation[np.newaxis]).float()
        }


class Simple2DTransformDataset(Dataset):
    def __init__(self, split='train'):
        super().__init__()
        assert split in ['train', 'valid'], f'Split parameters "{split}" must be either "train" or "valid".'
        # Read either train or validation data from disk based on split parameter.
        # Data is located in the folder "data".
        # Hint: you can use os.path.join to obtain a path in a subfolder.
        # Save samples and annotations to class members self.samples and self.annotations respectively.
        # Samples should be an Nx2 numpy array. Annotations should be Nx1.
        path="data"
        if split == "train":
            data = np.load(os.path.join(path, "", "train.npz"))
        elif split == "valid":
            data =  np.load(os.path.join(path, "", "valid.npz"))
        self.samples = data['samples']
        self.annotations = data['annotations']
            
    def __len__(self):
        # Returns the number of samples in the dataset.
        return self.samples.shape[0]
    
    def __getitem__(self, idx):
        # Returns the sample and annotation with index idx.
        sample = self.samples[idx]
        annotation = self.annotations[idx]
        
        # Transform the sample to a different coordinate system.
        sample = transform(sample)

        # Convert to tensor.
        return {
            'input': torch.from_numpy(sample).float(),
            'annotation': torch.from_numpy(annotation[np.newaxis]).float()
        }


def transform(sample):
    # Takes the x and y coordinates and computes their correspodning r and theta coordinates
    r = np.sqrt(sample[0]*sample[0]+sample[1]*sample[1])
    theta = np.arctan2(sample[0],sample[1])
    new_sample = np.array([r,theta])
    return new_sample
