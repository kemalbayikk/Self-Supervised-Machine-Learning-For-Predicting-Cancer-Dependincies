import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os
import pickle
import time
import wandb
from datetime import datetime

def load_data(filename):
    data = []
    gene_names = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        sample_names = lines[0].strip().split('\t')[1:]

        for line in lines[1:]:
            values = line.strip().split('\t')
            gene = values[0].upper()
            gene_names.append(gene)
            data.append(values[1:])

    data = np.array(data, dtype='float32').T
    return torch.tensor(data, dtype=torch.float32), sample_names, gene_names

if __name__ == '__main__':
    omic = "mut"

    for i in range(1,6):

        filepath = f"Data/TCGA/tcga_{omic}_data_paired_with_ccl.txt"
        data_tcga, sample_names_tcga, gene_names_tcga = load_data(filepath)

        # Split the data into training, validation, and test sets
        dataset = TensorDataset(data_tcga)
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        print(len(train_dataset))
        print(len(val_dataset))
        print(len(test_dataset))

        # Save train, validation, and test datasets
        torch.save(train_dataset, f'PytorchStaticSplits/TCGASplits/split_{i}/train_dataset_{omic}_split_{i}.pth')
        torch.save(val_dataset, f'PytorchStaticSplits/TCGASplits/split_{i}/val_dataset_{omic}_split_{i}.pth')
        torch.save(test_dataset, f'PytorchStaticSplits/TCGASplits/split_{i}/test_dataset_{omic}_split_{i}.pth')
        print('Datasets saved: train_dataset.pth, val_dataset.pth, test_dataset.pth')
