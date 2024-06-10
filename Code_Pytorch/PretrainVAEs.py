import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
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

    data = np.array(data, dtype='float32').T
    return torch.tensor(data, dtype=torch.float32), sample_names, gene_names

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, first_layer_dim, second_layer_dim, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, first_layer_dim)
        self.fc2 = nn.Linear(first_layer_dim, second_layer_dim)
        self.fc31 = nn.Linear(second_layer_dim, latent_dim)
        self.fc32 = nn.Linear(second_layer_dim, latent_dim)
        self.fc4 = nn.Linear(latent_dim, second_layer_dim)
        self.fc5 = nn.Linear(second_layer_dim, first_layer_dim)
        self.fc6 = nn.Linear(first_layer_dim, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc4(z))
        h4 = torch.relu(self.fc5(h3))
        return self.fc6(h4)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return MSE + KLD

def save_weights_to_pickle(model, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    weights = {name: param.to('cpu').detach().numpy() for name, param in model.named_parameters()}
    with open(file_name, 'wb') as handle:
        pickle.dump(weights, handle)
    print(f"Model weights saved to {file_name}")

if __name__ == '__main__':
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    filepath = "Data/TCGA/tcga_mut_data_paired_with_ccl.txt"
    data_mut_tcga, sample_names_mut_tcga, gene_names_mut_tcga = load_data(filepath)
    data_mut_tcga = data_mut_tcga.to(device)

    # Split the data into training and validation sets
    dataset = TensorDataset(data_mut_tcga)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = VariationalAutoencoder(input_dim=data_mut_tcga.shape[1], first_layer_dim=500, second_layer_dim=200, latent_dim=50)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 100
    patience = 10
    best_loss = float('inf')
    early_stop_counter = 0

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for data in train_loader:
            inputs = data[0].to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(inputs)
            loss = loss_function(recon_batch, inputs, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                inputs = data[0].to(device)
                recon_batch, mu, logvar = model(inputs)
                loss = loss_function(recon_batch, inputs, mu, logvar)
                val_loss += loss.item()
        val_loss /= len(val_loader.dataset)

        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.8f}, Validation Loss: {val_loss:.8f}')

        # # Early stopping
        # if val_loss < best_loss:
        #     best_loss = val_loss
        #     early_stop_counter = 0
        #     # Save the model's best weights
        #     save_weights_to_pickle(model, './results/autoencoders/premodel_tcga_mut_vae_best.pickle')
        # else:
        #     early_stop_counter += 1
        #     if early_stop_counter >= patience:
        #         print(f'Early stopping at epoch {epoch + 1}')
        #         break

    print('\nVAE training completed in %.1f mins' % ((time.time() - start_time) / 60))

    model_save_name = 'premodel_tcga_mut_vae_500_200_50.pickle'
    save_weights_to_pickle(model, './results/autoencoders/' + model_save_name)
    print("\nResults saved in /results/autoencoders/%s\n\n" % model_save_name)
