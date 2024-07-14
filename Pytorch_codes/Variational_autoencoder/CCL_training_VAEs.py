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

    def reparameterize(self, mu, logvar, var_scale=1.0):
        std = torch.exp(0.5 * logvar * var_scale)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc4(z))
        h4 = torch.relu(self.fc5(h3))
        return self.fc6(h4)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar, data_name, recon_weight=1.0, kl_weight=1.0):
    if data_name == "mut" or data_name == "fprint":
        recon_loss = recon_weight * nn.functional.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
    else:
        recon_loss = recon_weight * nn.functional.mse_loss(recon_x, x, reduction='sum')

    kl_loss = kl_weight * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss

def train_vae(model, train_loader, test_loader, num_epochs, learning_rate, device, data_name):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data in tqdm(train_loader):
            inputs = data[0].to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(inputs)
            loss, _, _ = vae_loss_function(recon_batch, inputs, mu, logvar, data_name)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        train_loss /= len(train_loader.dataset)


        model.eval()
        test_loss = 0
        test_recon_loss = 0
        test_kl_loss = 0
        with torch.no_grad():
            for data in test_loader:
                inputs = data[0].to(device)
                recon_batch, mu, logvar = model(inputs)
                loss, recon_loss, kl_loss = vae_loss_function(recon_batch, inputs, mu, logvar, data_name)
                test_loss += loss.item()
                test_recon_loss += recon_loss.item()
                test_kl_loss += kl_loss.item()
        test_loss /= len(test_loader.dataset)
        test_recon_loss /= len(test_loader.dataset) 
        test_kl_loss /= len(test_loader.dataset) 

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, Recon Loss: {test_recon_loss:.6f}, KL Loss: {test_kl_loss:.6f}')

        wandb.log({
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_recon_loss": test_recon_loss,
            "test_kl_loss": test_kl_loss
        })

    return model

def save_weights_to_pickle(model, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    weights = {name: param.to('cpu').detach().numpy() for name, param in model.named_parameters()}
    with open(file_name, 'wb') as handle:
        pickle.dump(weights, handle)
    print(f"Model weights saved to {file_name}")

def load_pretrained_vae(filepath, input_dim, first_layer_dim, second_layer_dim, latent_dim):
    vae = VariationalAutoencoder(input_dim, first_layer_dim, second_layer_dim, latent_dim)
    vae_state = pickle.load(open(filepath, 'rb'))

    # Convert numpy arrays to PyTorch tensors
    for key in vae_state:
        if isinstance(vae_state[key], np.ndarray):
            vae_state[key] = torch.tensor(vae_state[key])

    vae.load_state_dict(vae_state)
    return vae

if __name__ == '__main__':
    with open('Data/ccl_complete_data_278CCL_1298DepOI_360844samples.pickle', 'rb') as f:
        data_mut, data_exp, data_cna, data_meth, data_dep, data_fprint = pickle.load(f)
    # with open('Data/ccl_complete_data_28CCL_1298DepOI_36344samples_demo.pickle', 'rb') as f:
    #     data_mut, data_exp, data_cna, data_meth, data_dep, data_fprint = pickle.load(f)

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

        run = wandb.init(project="Self-Supervised-Machine-Learning-For-Predicting-Cancer-Dependencies", entity="kemal-bayik", name=f"SL_{data_type}_{ccl_size}CCL_{current_time}_500Batch")

        config = wandb.config
        config.learning_rate = 1e-4
        config.batch_size = 500
        config.epochs = epochs

        # Split the data into training and validation sets
        dataset = TensorDataset(tensor_data_ccl)
        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

        # Define model dimensions and load pretrained VAEs
        if data_type == 'mut':
            vae = load_pretrained_vae('results/variational_autoencoders/USL_pretrained/premodel_tcga_mut_vae__64batch_best.pickle', tensor_data_ccl.shape[1], 1000, 100, 50)
        elif data_type == 'exp':
            vae = load_pretrained_vae('results/variational_autoencoders/USL_pretrained/premodel_tcga_exp_vae__64batch_best.pickle', tensor_data_ccl.shape[1], 500, 200, 50)
        elif data_type == 'cna':
            vae = load_pretrained_vae('results/variational_autoencoders/USL_pretrained/premodel_tcga_cna_vae__64batch_best.pickle', tensor_data_ccl.shape[1], 500, 200, 50)
        elif data_type == 'meth':
            vae = load_pretrained_vae('results/variational_autoencoders/USL_pretrained/premodel_tcga_meth_vae__64batch_best.pickle', tensor_data_ccl.shape[1], 500, 200, 50)
        elif data_type == 'fprint':
            vae = VariationalAutoencoder(input_dim=tensor_data_ccl.shape[1], first_layer_dim=1000, second_layer_dim=100, latent_dim=50)
        
        # Train VAE
        trained_vae = train_vae(vae, train_loader, test_loader, num_epochs=config.epochs, learning_rate=config.learning_rate, device=device, data_name=data_type)

        wandb.log({
            "learning_rate": config.learning_rate,
            "batch_size": train_loader.batch_size,
            "epoch": epochs
        })
        
        # Save model weights
        save_weights_to_pickle(trained_vae, f'./results/variational_autoencoders/premodel_ccl_{data_type}_vae_64batch.pickle')
        run.finish()
