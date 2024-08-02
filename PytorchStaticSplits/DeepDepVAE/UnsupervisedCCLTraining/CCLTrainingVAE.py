import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import pickle
import time
from tqdm import tqdm
import wandb
from datetime import datetime

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
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

def vae_loss_function(recon_x, x, mu, logvar, data_name, beta):
    if data_name == "mut" or data_name == "fprint":
        recon_loss = nn.functional.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
    else:
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')

    kl_loss = beta * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss

def train_vae(model, train_loader, val_loader, num_epochs, learning_rate, device, data_name, beta):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    best_loss = float('inf')
    early_stop_counter = 0
    patience = 10

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data in tqdm(train_loader):
            inputs = data[0].to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(inputs)
            loss, _, _ = vae_loss_function(recon_batch, inputs, mu, logvar, data_name, beta)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        train_loss /= len(train_loader.dataset)


        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kl_loss = 0
        with torch.no_grad():
            for data in val_loader:
                inputs = data[0].to(device)
                recon_batch, mu, logvar = model(inputs)
                loss, recon_loss, kl_loss = vae_loss_function(recon_batch, inputs, mu, logvar, data_name, beta)
                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_kl_loss += kl_loss.item()
        val_loss /= len(val_loader.dataset)
        val_recon_loss /= len(val_loader.dataset) 
        val_kl_loss /= len(val_loader.dataset) 

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Recon Loss: {val_recon_loss:.6f}, KL Loss: {val_kl_loss:.6f}')

        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_recon_loss": val_recon_loss,
            "val_kl_loss": val_kl_loss,
            "beta": beta
        })

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            early_stop_counter = 0
            # Save the model's best weights
            save_weights_to_pickle(model, f'PytorchStaticSplits/DeepDepVAE/Results/Split{split_num}/CCL_Pretrained/ccl_{data_name}_vae_best_split_{split_num}_Beta1_LR_Test_After.pickle')

    return model

def evaluate_vae(model, test_loader, device, data_name, beta):
    model.eval()
    test_loss = 0
    test_recon_loss = 0
    test_kl_loss = 0
    with torch.no_grad():
        for data in test_loader:
            inputs = data[0].to(device)
            recon_batch, mu, logvar = model(inputs)
            loss, recon_loss, kl_loss = vae_loss_function(recon_batch, inputs, mu, logvar, data_name, beta)
            test_loss += loss.item()
            test_recon_loss += recon_loss.item()
            test_kl_loss += kl_loss.item()
    test_loss /= len(test_loader.dataset)
    test_recon_loss /= len(test_loader.dataset)
    test_kl_loss /= len(test_loader.dataset)

    wandb.log({
        "test_loss": test_loss,
        "test_recon_loss": test_recon_loss,
        "test_kl_loss": test_kl_loss,
        "beta": beta
    })

    print(f'Test Loss: {test_loss:.6f}, Test Recon Loss: {test_recon_loss:.6f}, Test KL Loss: {test_kl_loss:.6f}')

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
    ccl_size = 278
    epochs = 100
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # data_types = ['mut', 'exp', 'cna', 'meth', 'fprint']
    
    for split_num in range(1, 6):
        with open(f'Data/data_split_{split_num}.pickle', 'rb') as f:
            train_dataset, val_dataset, test_dataset = pickle.load(f)
        
        data_dict = {
            'mut': {
                'train':train_dataset[:][0],
                'val':val_dataset[:][0],
                'test':test_dataset[:][0]
                },  
            'exp': {
                'train':train_dataset[:][1],
                'val':val_dataset[:][1],
                'test':test_dataset[:][1]
                },
            'cna': {
                'train':train_dataset[:][2],
                'val':val_dataset[:][2],
                'test':test_dataset[:][2]
                },
            'meth': {
                'train':train_dataset[:][3],
                'val':val_dataset[:][3],
                'test':test_dataset[:][3]
                },
            'fprint': {
                'train':train_dataset[:][4],
                'val':val_dataset[:][4],
                'test':test_dataset[:][4]
                },
        }

        for data_type, data_ccl in data_dict.items():

            print(data_ccl["train"])
            run = wandb.init(project="DeepDepVAELRTest", entity="kemal-bayik", name=f"SL_{data_type}_{ccl_size}CCL_{current_time}_Split{split_num}_Beta1_LR_Test_After")

            config = wandb.config
            config.learning_rate = 1e-4
            config.batch_size = 10000
            config.epochs = epochs
            config.beta = 1

            # Extract tensors from val_dataset and test_dataset
            train_tensors = torch.tensor(data_ccl["train"], dtype=torch.float32).to(device)
            val_tensors = torch.tensor(data_ccl["val"], dtype=torch.float32).to(device)
            test_tensors = torch.tensor(data_ccl["test"], dtype=torch.float32).to(device)

            print(len(data_ccl["train"]))
            print(len(data_ccl["val"]))
            print(len(data_ccl["test"]))

            # Create DataLoader for train, val, and test sets
            train_loader = DataLoader(TensorDataset(train_tensors), batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(val_tensors), batch_size=config.batch_size, shuffle=False)
            test_loader = DataLoader(TensorDataset(test_tensors), batch_size=config.batch_size, shuffle=False)

            # Define model dimensions and load pretrained VAEs
            if data_type == 'mut':
                vae = load_pretrained_vae(f'PytorchStaticSplits/DeepDepVAE/Results/Split{split_num}/USL_pretrained/tcga_mut_vae_best_split_{split_num}_Beta1_LR3.pickle', train_tensors.shape[1], 1000, 100, 50)
            elif data_type == 'exp':
                vae = load_pretrained_vae(f'PytorchStaticSplits/DeepDepVAE/Results/Split{split_num}/USL_Pretrained/tcga_exp_vae_best_split_{split_num}.pickle', train_tensors.shape[1], 500, 200, 50)
            elif data_type == 'cna':
                vae = load_pretrained_vae(f'PytorchStaticSplits/DeepDepVAE/Results/Split{split_num}/USL_Pretrained/tcga_cna_vae_best_split_{split_num}.pickle', train_tensors.shape[1], 500, 200, 50)
            elif data_type == 'meth':
                vae = load_pretrained_vae(f'PytorchStaticSplits/DeepDepVAE/Results/Split{split_num}/USL_Pretrained/tcga_meth_vae_best_split_{split_num}_Beta1_LR3.pickle', train_tensors.shape[1], 500, 200, 50)
            elif data_type == 'fprint':
                vae = VariationalAutoencoder(input_dim=train_tensors.shape[1], first_layer_dim=1000, second_layer_dim=100, latent_dim=50)
            
            # Train VAE
            trained_vae = train_vae(vae, train_loader, val_loader, num_epochs=config.epochs, learning_rate=config.learning_rate, device=device, data_name=data_type, beta=config.beta)

            # Evaluate VAE on test set
            evaluate_vae(trained_vae, test_loader, device, data_name=data_type, beta=config.beta)

            wandb.log({
                "learning_rate": config.learning_rate,
                "batch_size": train_loader.batch_size,
                "epoch": epochs
            })
        
        # Save model weights
        # save_weights_to_pickle(trained_vae, f'PytorchStaticSplits/DeepDepVAE/Results/Split{split_num}/CCL_Pretrained/premodel_ccl_{data_type}_vae_split_{split_num}.pickle')
            run.finish()
