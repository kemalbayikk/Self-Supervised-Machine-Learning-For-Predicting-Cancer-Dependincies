import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os
import pickle
import time
from tqdm import tqdm
import wandb
from datetime import datetime

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#device = "cuda"
print(device)

class LadderVariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, first_layer_dim, second_layer_dim, latent_dim, ladder_dim):
        super(LadderVariationalAutoencoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, first_layer_dim)
        self.bn1 = nn.BatchNorm1d(first_layer_dim)
        self.fc2 = nn.Linear(first_layer_dim, second_layer_dim)
        self.bn2 = nn.BatchNorm1d(second_layer_dim)
        self.fc31 = nn.Linear(second_layer_dim, latent_dim)
        self.fc32 = nn.Linear(second_layer_dim, latent_dim)
        self.fc4 = nn.Linear(latent_dim, second_layer_dim)
        self.bn4 = nn.BatchNorm1d(second_layer_dim)
        self.fc5 = nn.Linear(second_layer_dim, first_layer_dim)
        self.bn5 = nn.BatchNorm1d(first_layer_dim)
        self.fc6 = nn.Linear(first_layer_dim, input_dim)

        self.ladder_dim = ladder_dim
        self.ladder = nn.ModuleList([nn.Linear(latent_dim, ladder_dim) for _ in range(2)])
        self.fc7 = nn.Linear(ladder_dim, latent_dim)

    def encode(self, x):
        h1 = torch.relu(self.bn1(self.fc1(x)))
        h2 = torch.relu(self.bn2(self.fc2(h1)))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar, var_scale=1.0):
        std = torch.exp(0.5 * logvar * var_scale)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.bn4(self.fc4(z)))
        h4 = torch.relu(self.bn5(self.fc5(h3)))
        return self.fc6(h4)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        ladder_output = z
        for layer in self.ladder:
            ladder_output = torch.relu(layer(ladder_output))
        z_corrected = self.fc7(ladder_output)

        recon_x = self.decode(z_corrected)
        return recon_x, mu, logvar

def lvae_loss_function(recon_x, x, mu, logvar, data_name, beta, recon_weight=1.0, kl_weight=1.0):
    if data_name == "mut" or data_name == "fprint":
        recon_loss = recon_weight * nn.functional.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
    else:
        recon_loss = recon_weight * nn.functional.mse_loss(recon_x, x, reduction='sum')

    kl_loss = kl_weight * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

def train_lvae(model, train_loader, test_loader, num_epochs, learning_rate, device, data_name, beta_start, beta_end, omic):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    
    beta = beta_start
    beta_increment_per_epoch = (beta_end - beta_start) / (0.1 * num_epochs)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data in tqdm(train_loader):
            inputs = data[0].to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(inputs)
            loss, _, _ = lvae_loss_function(recon_batch, inputs, mu, logvar, data_name, beta)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        train_loss /= len(train_loader.dataset)

        if epoch < 0.1 * num_epochs:
            beta += beta_increment_per_epoch
        else:
            beta = beta_end

        model.eval()
        test_loss = 0
        test_recon_loss = 0
        test_kl_loss = 0
        with torch.no_grad():
            for data in test_loader:
                inputs = data[0].to(device)
                recon_batch, mu, logvar = model(inputs)
                loss, recon_loss, kl_loss = lvae_loss_function(recon_batch, inputs, mu, logvar, data_name, beta)
                test_loss += loss.item()
                test_recon_loss += recon_loss.item()
                test_kl_loss += kl_loss.item()
        test_loss /= len(test_loader.dataset)
        test_recon_loss /= len(test_loader.dataset) 
        test_kl_loss /= len(test_loader.dataset) 

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, Recon Loss: {test_recon_loss:.6f}, KL Loss: {test_kl_loss:.6f}')

        if test_loss < best_loss:
            best_loss = test_loss
            # Save the model's best weights
            save_weights_to_pickle(model, f'./results/ladder_variational_autoencoders/premodel_ccl_{omic}_vae_best_lvae.pickle')

        wandb.log({
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_recon_loss": test_recon_loss,
            "test_kl_loss": test_kl_loss,
            "beta": beta
        })

    return model

def save_weights_to_pickle(model, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'wb') as handle:
        pickle.dump(model.state_dict(), handle)
    print(f"Model weights saved to {file_name}")

def load_pretrained_lvae(filepath, input_dim, first_layer_dim, second_layer_dim, latent_dim, ladder_dim):
    lvae = LadderVariationalAutoencoder(input_dim, first_layer_dim, second_layer_dim, latent_dim, ladder_dim)
    with open(filepath, 'rb') as handle:
        lvae_state = pickle.load(handle)

    # Convert numpy arrays to PyTorch tensors
    for key in lvae_state:
        if isinstance(lvae_state[key], np.ndarray):
            lvae_state[key] = torch.tensor(lvae_state[key])

    lvae.load_state_dict(lvae_state)
    return lvae

if __name__ == '__main__':
    with open('Data/ccl_complete_data_278CCL_1298DepOI_360844samples.pickle', 'rb') as f:
        data_mut, data_exp, data_cna, data_meth, data_dep, data_fprint = pickle.load(f)

    ccl_size = 278
    epochs = 100
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    data_dict = {
        'mut': data_mut,
        'exp': data_exp,
        'cna': data_cna,
        'meth': data_meth,
        'fprint': data_fprint
    }

    for data_type, data_ccl in data_dict.items():
        tensor_data_ccl = torch.tensor(data_ccl, dtype=torch.float32).to(device)

        run = wandb.init(project="Self-Supervised-Machine-Learning-For-Predicting-Cancer-Dependencies", entity="kemal-bayik", name=f"SL_{data_type}_{ccl_size}CCL_{current_time}_LVAE")

        config = wandb.config
        config.learning_rate = 1e-3
        config.batch_size = 500
        config.epochs = epochs
        config.beta_start = 0.0
        config.beta_end = 1.0

        # Split the data into training and validation sets
        dataset = TensorDataset(tensor_data_ccl)
        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

        if data_type == 'mut':
            lvae = load_pretrained_lvae('results/ladder_variational_autoencoders/USL_pretrained/premodel_tcga_mut_lvae_best.pickle', tensor_data_ccl.shape[1], 1000, 100, 50, 50)
        elif data_type == 'exp':
            lvae = load_pretrained_lvae('results/ladder_variational_autoencoders/USL_pretrained/premodel_tcga_exp_lvae_best.pickle', tensor_data_ccl.shape[1], 500, 200, 50, 50)
        elif data_type == 'cna':
            lvae = load_pretrained_lvae('results/ladder_variational_autoencoders/USL_pretrained/premodel_tcga_cna_lvae_best.pickle', tensor_data_ccl.shape[1], 500, 200, 50, 50)
        elif data_type == 'meth':
            lvae = load_pretrained_lvae('results/ladder_variational_autoencoders/USL_pretrained/premodel_tcga_meth_lvae_best.pickle', tensor_data_ccl.shape[1], 500, 200, 50, 50)
        elif data_type == 'fprint':
            lvae = LadderVariationalAutoencoder(input_dim=tensor_data_ccl.shape[1], first_layer_dim=1000, second_layer_dim=100, latent_dim=50, ladder_dim=50)
        
        # Train LVAE
        trained_lvae = train_lvae(lvae, train_loader, test_loader, num_epochs=config.epochs, learning_rate=config.learning_rate, device=device, data_name=data_type, beta_start=config.beta_start, beta_end=config.beta_end, omic=data_type)

        wandb.log({
            "learning_rate": config.learning_rate,
            "batch_size": train_loader.batch_size,
            "epoch": epochs
        })
        
        # Save model weights
        save_weights_to_pickle(trained_lvae, f'./results/ladder_variational_autoencoders/premodel_ccl_{data_type}_lvae.pickle')
        run.finish()
