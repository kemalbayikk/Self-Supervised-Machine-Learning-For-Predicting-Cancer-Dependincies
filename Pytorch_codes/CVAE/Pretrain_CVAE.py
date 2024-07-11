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

class ConditionalVariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, cond_dim, first_layer_dim, second_layer_dim, latent_dim):
        super(ConditionalVariationalAutoencoder, self).__init__()
        self.fc1 = nn.Linear(input_dim + cond_dim, first_layer_dim)
        self.fc2 = nn.Linear(first_layer_dim, second_layer_dim)
        self.fc31 = nn.Linear(second_layer_dim, latent_dim)
        self.fc32 = nn.Linear(second_layer_dim, latent_dim)
        self.fc4 = nn.Linear(latent_dim + cond_dim, second_layer_dim)
        self.fc5 = nn.Linear(second_layer_dim, first_layer_dim)
        self.fc6 = nn.Linear(first_layer_dim, input_dim)

    def encode(self, x, c):
        xc = torch.cat([x, c], dim=1)
        h1 = torch.relu(self.fc1(xc))
        h2 = torch.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        zc = torch.cat([z, c], dim=1)
        h3 = torch.relu(self.fc4(zc))
        h4 = torch.relu(self.fc5(h3))
        return self.fc6(h4)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar, data_name, recon_weight=1.0, kl_weight=1.0):
    if data_name == "mut":
        recon_loss = recon_weight * nn.functional.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
    else:
        recon_loss = recon_weight * nn.functional.mse_loss(recon_x, x, reduction='sum')

    kl_loss = kl_weight * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss

def save_weights_to_pickle(model, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    weights = {name: param.to('cpu').detach().numpy() for name, param in model.named_parameters()}
    with open(file_name, 'wb') as handle:
        pickle.dump(weights, handle)
    print(f"Model weights saved to {file_name}")

if __name__ == '__main__':

    omics = ["mut", "cna", "exp", "meth"]

    for omic in omics: 
        
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        wandb.init(project="Self-Supervised-Machine-Learning-For-Predicting-Cancer-Dependencies", entity="kemal-bayik", name=f"TCGA_{omic}_{current_time}_CVAE")
        learning_rate = 1e-4
        batch_size = 10000
        epochs = 100
        patience = 10

        if omic == "mut":
            first_layer_dim = 1000
            second_layer_dim = 100
            latent_dim = 50
        else: 
            first_layer_dim = 500
            second_layer_dim = 200
            latent_dim = 50

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        filepath = f"Data/TCGA/tcga_{omic}_data_paired_with_ccl.txt"
        data_tcga, sample_names_tcga, gene_names_tcga = load_data(filepath)
        data_tcga = data_tcga.to(device)

        # Dummy conditional data, this should be replaced with your actual conditional data
        cond_data = torch.randn((data_tcga.shape[0], 10)).to(device)

        input_dim = data_tcga.shape[1]
        cond_dim = cond_data.shape[1]

        # Split the data into training and validation sets
        dataset = TensorDataset(data_tcga, cond_data)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = ConditionalVariationalAutoencoder(input_dim=input_dim, cond_dim=cond_dim, first_layer_dim=first_layer_dim, second_layer_dim=second_layer_dim, latent_dim=latent_dim)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        best_loss = float('inf')
        early_stop_counter = 0

        start_time = time.time()
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for data, cond in train_loader:
                inputs = data.to(device)
                cond = cond.to(device)
                optimizer.zero_grad()
                recon_batch, mu, logvar = model(inputs, cond)
                loss, recon_loss, kl_loss = vae_loss_function(recon_batch, inputs, mu, logvar, omic)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            train_loss /= len(train_loader.dataset)

            model.eval()
            val_loss = 0
            test_recon_loss = 0
            test_kl_loss = 0
            with torch.no_grad():
                for data, cond in val_loader:
                    inputs = data.to(device)
                    cond = cond.to(device)
                    recon_batch, mu, logvar = model(inputs, cond)
                    loss, recon_loss, kl_loss = vae_loss_function(recon_batch, inputs, mu, logvar, omic)
                    test_recon_loss += recon_loss.item()
                    test_kl_loss += kl_loss.item()
                    val_loss += loss.item()
            val_loss /= len(val_loader.dataset)
            test_recon_loss /= len(val_loader.dataset) 
            test_kl_loss /= len(val_loader.dataset) 

            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epoch": epoch + 1,
                "test_recon_loss": test_recon_loss,
                "test_kl_loss": test_kl_loss
            })

            print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}, Recon Loss: {test_recon_loss:.6f}, KL Loss: {test_kl_loss:.6f}')

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                early_stop_counter = 0
                # Save the model's best weights
                save_weights_to_pickle(model, f'./results/conditional_variational_autoencoders/USL_pretrained/premodel_tcga_{omic}_cvae_best.pickle')

        print('\nCVAE training completed in %.1f mins' % ((time.time() - start_time) / 60))

        model_save_name = f'premodel_tcga_{omic}_cvae.pickle'
        save_weights_to_pickle(model, './results/conditional_variational_autoencoders/USL_pretrained/' + model_save_name)
        print("\nResults saved in /results/conditional_variational_autoencoders/USL_pretrained/%s\n\n" % model_save_name)
