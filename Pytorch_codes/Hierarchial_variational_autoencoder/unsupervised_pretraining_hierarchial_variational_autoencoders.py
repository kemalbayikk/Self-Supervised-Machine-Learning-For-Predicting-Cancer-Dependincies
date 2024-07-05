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

class HierarchicalVariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, first_layer_dim, second_layer_dim, latent_dim1, latent_dim2):
        super(HierarchicalVariationalAutoencoder, self).__init__()
        # First level encoder
        self.fc1 = nn.Linear(input_dim, first_layer_dim)
        self.fc2 = nn.Linear(first_layer_dim, second_layer_dim)
        
        # Second level encoder (first latent space)
        self.fc31_mu = nn.Linear(second_layer_dim, latent_dim1)
        self.fc31_logvar = nn.Linear(second_layer_dim, latent_dim1)
        
        # Third level encoder (second latent space, hierarchical)
        self.fc32_mu = nn.Linear(latent_dim1, latent_dim2)
        self.fc32_logvar = nn.Linear(latent_dim1, latent_dim2)
        
        # First level decoder (from second latent space)
        self.fc4 = nn.Linear(latent_dim2, latent_dim1)
        
        # Second level decoder (from first latent space)
        self.fc5 = nn.Linear(latent_dim1, second_layer_dim)
        self.fc6 = nn.Linear(second_layer_dim, first_layer_dim)
        self.fc7 = nn.Linear(first_layer_dim, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        mu1 = self.fc31_mu(h2)
        logvar1 = self.fc31_logvar(h2)
        z1 = self.reparameterize(mu1, logvar1)
        mu2 = self.fc32_mu(z1)
        logvar2 = self.fc32_logvar(z1)
        return mu1, logvar1, mu2, logvar2

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z2):
        h3 = torch.relu(self.fc4(z2))
        h4 = torch.relu(self.fc5(h3))
        h5 = torch.relu(self.fc6(h4))
        return self.fc7(h5)

    def forward(self, x):
        mu1, logvar1, mu2, logvar2 = self.encode(x)
        z2 = self.reparameterize(mu2, logvar2)
        recon_x = self.decode(z2)
        return recon_x, mu1, logvar1, mu2, logvar2

def hvae_loss_function(recon_x, x, mu1, logvar1, mu2, logvar2, data_name, recon_weight=1.0, kl_weight=1.0):
    if data_name == "mut":
        recon_loss = recon_weight * nn.functional.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
    else:
        recon_loss = recon_weight * nn.functional.mse_loss(recon_x, x, reduction='sum')

    kl_loss1 = kl_weight * -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp())
    kl_loss2 = kl_weight * -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp())
    return recon_loss + kl_loss1 + kl_loss2, recon_loss, kl_loss1 + kl_loss2

def save_weights_to_pickle(model, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    weights = {name: param.to('cpu').detach().numpy() for name, param in model.named_parameters()}
    with open(file_name, 'wb') as handle:
        pickle.dump(weights, handle)
    print(f"Model weights saved to {file_name}")

if __name__ == '__main__':
    omic = "meth"
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb.init(project="Self-Supervised-Machine-Learning-For-Predicting-Cancer-Dependencies", entity="kemal-bayik", name=f"TCGA_{omic}_{current_time}_HVAE")

    config = wandb.config
    config.learning_rate = 1e-4
    config.batch_size = 500
    config.epochs = 100
    config.patience = 10
    config.first_layer_dim = 500
    config.second_layer_dim = 200
    config.latent_dim1 = 50
    config.latent_dim2 = 25

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    filepath = f"Data/TCGA/tcga_{omic}_data_paired_with_ccl.txt"
    data_tcga, sample_names_tcga, gene_names_tcga = load_data(filepath)
    data_tcga = data_tcga.to(device)

    config.input_dim = data_tcga.shape[1]

    # Split the data into training and validation sets
    dataset = TensorDataset(data_tcga)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model = HierarchicalVariationalAutoencoder(input_dim=data_tcga.shape[1], first_layer_dim=config.first_layer_dim, second_layer_dim=config.second_layer_dim, latent_dim1=config.latent_dim1, latent_dim2=config.latent_dim2)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    best_loss = float('inf')
    early_stop_counter = 0

    start_time = time.time()
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        for data in train_loader:
            inputs = data[0].to(device)
            optimizer.zero_grad()
            recon_batch, mu1, logvar1, mu2, logvar2 = model(inputs)
            loss, recon_loss, kl_loss = hvae_loss_function(recon_batch, inputs, mu1, logvar1, mu2, logvar2, omic)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        test_recon_loss = 0
        test_kl_loss = 0
        with torch.no_grad():
            for data in val_loader:
                inputs = data[0].to(device)
                recon_batch, mu1, logvar1, mu2, logvar2 = model(inputs)
                loss, recon_loss, kl_loss = hvae_loss_function(recon_batch, inputs, mu1, logvar1, mu2, logvar2, omic)
                val_loss += loss.item()
                test_recon_loss += recon_loss
                test_kl_loss += kl_loss
        val_loss /= len(val_loader.dataset)
        test_recon_loss /= len(val_loader.dataset) 
        test_kl_loss /= len(val_loader.dataset) 

        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "epoch": epoch + 1,
            "test_recon_loss": test_recon_loss,
            "test_kl_loss": test_kl_loss
        })

        print(f'Epoch [{epoch + 1}/{config.epochs}], Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}, Recon Loss: {test_recon_loss:.6f}, KL Loss: {test_kl_loss:.6f}')

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            early_stop_counter = 0
            # Save the model's best weights
            save_weights_to_pickle(model, f'./results/hierarchial_variational_autoencoders/USL_pretrained/premodel_tcga_{omic}_hvae_best.pickle')
        # else:
        #     early_stop_counter += 1
        #     if early_stop_counter >= config.patience:
        #         print(f'Early stopping at epoch {epoch + 1}')
        #         break

    print('\nHVAE training completed in %.1f mins' % ((time.time() - start_time) / 60))

    model_save_name = f'premodel_tcga_{omic}_hvae_500_200_50.pickle'
    save_weights_to_pickle(model, './results/hierarchial_variational_autoencoders/USL_pretrained/' + model_save_name)
    print("\nResults saved in /results/hierarchial_variational_autoencoder/USL_pretrained/%s\n\n" % model_save_name)
