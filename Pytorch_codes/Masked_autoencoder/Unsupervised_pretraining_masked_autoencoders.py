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

class MaskedAutoencoder(nn.Module):
    def __init__(self, input_dim, first_layer_dim, second_layer_dim, latent_dim):
        super(MaskedAutoencoder, self).__init__()
        self.encoder_fc1 = nn.Linear(input_dim, first_layer_dim)
        self.encoder_fc2 = nn.Linear(first_layer_dim, second_layer_dim)
        self.encoder_fc3 = nn.Linear(second_layer_dim, latent_dim)
        self.decoder_fc1 = nn.Linear(latent_dim, second_layer_dim)
        self.decoder_fc2 = nn.Linear(second_layer_dim, first_layer_dim)
        self.decoder_fc3 = nn.Linear(first_layer_dim, input_dim)

    def forward(self, x, mask_ratio=0.75):
        mask = torch.rand(x.shape).to(x.device) < mask_ratio
        x_masked = x * mask.float()

        encoded = torch.relu(self.encoder_fc1(x_masked))
        encoded = torch.relu(self.encoder_fc2(encoded))
        latent = self.encoder_fc3(encoded)

        decoded = torch.relu(self.decoder_fc1(latent))
        decoded = torch.relu(self.decoder_fc2(decoded))
        reconstructed = self.decoder_fc3(decoded)

        return reconstructed, mask

def mae_loss_function(recon_x, x, mask, data_name):
    if data_name == "mut":
        loss = nn.functional.binary_cross_entropy_with_logits(recon_x * mask.float(), x * mask.float(), reduction='sum') / mask.float().sum()
    else:
        loss = nn.functional.mse_loss(recon_x * mask.float(), x * mask.float(), reduction='sum') / mask.float().sum()
    return loss

def save_weights_to_pickle(model, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    weights = {name: param.to('cpu').detach().numpy() for name, param in model.named_parameters()}
    with open(file_name, 'wb') as handle:
        pickle.dump(weights, handle)
    print(f"Model weights saved to {file_name}")

if __name__ == '__main__':
    omic = "meth"
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb.init(project="Self-Supervised-Machine-Learning-For-Predicting-Cancer-Dependencies", entity="kemal-bayik", name=f"TCGA_{omic}_{current_time}_MAE")

    config = wandb.config
    config.learning_rate = 1e-4
    config.batch_size = 10000
    config.epochs = 100
    config.patience = 10
    config.first_layer_dim = 500
    config.second_layer_dim = 200
    config.latent_dim = 50

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

    model = MaskedAutoencoder(input_dim=data_tcga.shape[1], first_layer_dim=config.first_layer_dim, second_layer_dim=config.second_layer_dim, latent_dim=config.latent_dim)
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
            recon_batch, mask = model(inputs)
            loss = mae_loss_function(recon_batch, inputs, mask, omic)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        test_recon_loss = 0
        with torch.no_grad():
            for data in val_loader:
                inputs = data[0].to(device)
                recon_batch, mask = model(inputs)
                loss = mae_loss_function(recon_batch, inputs, mask, omic)
                val_loss += loss.item()
                test_recon_loss += loss.item()
        val_loss /= len(val_loader.dataset)
        test_recon_loss /= len(val_loader.dataset)

        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "epoch": epoch + 1,
            "test_recon_loss": test_recon_loss
        })

        print(f'Epoch [{epoch + 1}/{config.epochs}], Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}, Recon Loss: {test_recon_loss:.6f}')

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            early_stop_counter = 0
            # Save the model's best weights
            save_weights_to_pickle(model, f'./results/masked_autoencoders/USL_pretrained/premodel_tcga_{omic}_mae_best.pickle')

    print('\nMAE training completed in %.1f mins' % ((time.time() - start_time) / 60))

    model_save_name = f'premodel_tcga_{omic}_mae.pickle'
    save_weights_to_pickle(model, './results/masked_autoencoders/USL_pretrained/' + model_save_name)
    print("\nResults saved in /results/masked_autoencoders/USL_pretrained/%s\n\n" % model_save_name)
