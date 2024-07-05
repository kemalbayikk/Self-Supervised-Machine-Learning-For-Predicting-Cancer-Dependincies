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
    if data_name == "mut" or data_name == "fprint":
        recon_loss = recon_weight * nn.functional.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
    else:
        recon_loss = recon_weight * nn.functional.mse_loss(recon_x, x, reduction='sum')

    kl_loss1 = kl_weight * -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp())
    kl_loss2 = kl_weight * -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp())
    return recon_loss + kl_loss1 + kl_loss2, recon_loss, kl_loss1 + kl_loss2

def train_hvae(model, train_loader, test_loader, num_epochs, learning_rate, device, data_name):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data in tqdm(train_loader):
            inputs = data[0].to(device)
            optimizer.zero_grad()
            recon_batch, mu1, logvar1, mu2, logvar2 = model(inputs)
            loss, _, _ = hvae_loss_function(recon_batch, inputs, mu1, logvar1, mu2, logvar2, data_name)
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
                recon_batch, mu1, logvar1, mu2, logvar2 = model(inputs)
                loss, recon_loss, kl_loss = hvae_loss_function(recon_batch, inputs, mu1, logvar1, mu2, logvar2, data_name)
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

def load_pretrained_hvae(filepath, input_dim, first_layer_dim, second_layer_dim, latent_dim1, latent_dim2):
    hvae = HierarchicalVariationalAutoencoder(input_dim, first_layer_dim, second_layer_dim, latent_dim1, latent_dim2)
    hvae_state = pickle.load(open(filepath, 'rb'))

    # Convert numpy arrays to PyTorch tensors
    for key in hvae_state:
        if isinstance(hvae_state[key], np.ndarray):
            hvae_state[key] = torch.tensor(hvae_state[key])

    hvae.load_state_dict(hvae_state)
    return hvae

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

        run = wandb.init(project="Self-Supervised-Machine-Learning-For-Predicting-Cancer-Dependencies", entity="kemal-bayik", name=f"SL_{data_type}_{ccl_size}CCL_{current_time}_HVAE")

        config = wandb.config
        config.learning_rate = 1e-4
        config.batch_size = 5000
        config.epochs = epochs

        # Split the data into training and validation sets
        dataset = TensorDataset(tensor_data_ccl)
        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

        # # Define model dimensions and load pretrained HVAEs
        if data_type == 'mut':
            hvae = load_pretrained_hvae('results/hierarchial_variational_autoencoders/USL_pretrained/premodel_tcga_mut_hvae_best.pickle', tensor_data_ccl.shape[1], 1000, 100, 50, 25)
        elif data_type == 'exp':
            hvae = load_pretrained_hvae('results/hierarchial_variational_autoencoders/USL_pretrained/premodel_tcga_exp_hvae_best.pickle', tensor_data_ccl.shape[1], 500, 200, 50, 25)
        elif data_type == 'cna':
            hvae = load_pretrained_hvae('results/hierarchial_variational_autoencoders/USL_pretrained/premodel_tcga_cna_hvae_best.pickle', tensor_data_ccl.shape[1], 500, 200, 50, 25)
        elif data_type == 'meth':
            hvae = load_pretrained_hvae('results/hierarchial_variational_autoencoders/USL_pretrained/premodel_tcga_meth_hvae_best.pickle', tensor_data_ccl.shape[1], 500, 200, 50, 25)
        elif data_type == 'fprint':
            hvae = HierarchicalVariationalAutoencoder(input_dim=tensor_data_ccl.shape[1], first_layer_dim=1000, second_layer_dim=100, latent_dim1=50, latent_dim2=25)

        # Train HVAE
        trained_hvae = train_hvae(hvae, train_loader, test_loader, num_epochs=config.epochs, learning_rate=config.learning_rate, device=device, data_name=data_type)

        wandb.log({
            "learning_rate": config.learning_rate,
            "batch_size": train_loader.batch_size,
            "epoch": epochs
        })

        # Save model weights
        save_weights_to_pickle(trained_hvae, f'./results/hierarchial_variational_autoencoders/premodel_ccl_{data_type}_hvae.pickle')
        run.finish()
