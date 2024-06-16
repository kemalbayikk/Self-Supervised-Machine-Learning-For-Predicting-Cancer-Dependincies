import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import pickle
import time

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

    data = np.array(data, dtype='float32').T  # Transpose to match PyTorch's batch-first expectation
    return torch.tensor(data), sample_names, gene_names

class Autoencoder(nn.Module):
    def __init__(self, input_dim, first_layer_dim, second_layer_dim, third_layer_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, first_layer_dim),
            nn.ReLU(True),
            nn.Linear(first_layer_dim, second_layer_dim),
            nn.ReLU(True),
            nn.Linear(second_layer_dim, third_layer_dim),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(third_layer_dim, second_layer_dim),
            nn.ReLU(True),
            nn.Linear(second_layer_dim, first_layer_dim),
            nn.ReLU(True),
            nn.Linear(first_layer_dim, input_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def save_weights_to_pickle(model, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    weights = {name: param.to('cpu').detach().numpy() for name, param in model.named_parameters()}
    with open(file_name, 'wb') as handle:
        pickle.dump(weights, handle)
    print(f"Model weights saved to {file_name}")


if __name__ == '__main__':
    # Set the device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    filepath = "Data/TCGA/tcga_mut_data_paired_with_ccl.txt"
    data_mut_tcga, sample_names_mut_tcga, gene_names_mut_tcga = load_data(filepath)
    data_mut_tcga = data_mut_tcga.to(device)

    dataset = TensorDataset(data_mut_tcga)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = Autoencoder(input_dim=data_mut_tcga.shape[1], first_layer_dim=1000, second_layer_dim=100, third_layer_dim=50)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 100
    start_time = time.time()
    for epoch in range(epochs):
        for data in dataloader:
            inputs, = data
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    cost = loss.item()
    print('\nAutoencoder training completed in %.1f mins with test loss:%.4f' % ((time.time() - start_time)/60, cost))

    model_save_name = 'premodel_tcga_mut_1000_100_50.pickle'
    save_weights_to_pickle(model, './results/autoencoders/' + model_save_name)
    print("\nResults saved in /results/autoencoders/%s\n\n" % model_save_name)
