import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import pickle
import time
import wandb
from datetime import datetime

def load_data(filename):
    data = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        sample_names = lines[0].strip().split('\t')[1:]

        for line in lines[1:]:
            values = line.strip().split('\t')[1:]
            data.append(values)

    data = np.array(data, dtype='float32')
    return torch.tensor(data, dtype=torch.float32)

def load_split(split_num, omic, base_path=''):
    train_data = torch.load(os.path.join(base_path, f'PytorchStaticSplits/TCGASplits/split_{split_num}/train_dataset_{omic}_split_{split_num}.pth'))
    val_data = torch.load(os.path.join(base_path, f'PytorchStaticSplits/TCGASplits/split_{split_num}/val_dataset_{omic}_split_{split_num}.pth'))
    test_data = torch.load(os.path.join(base_path, f'PytorchStaticSplits/TCGASplits/split_{split_num}/test_dataset_{omic}_split_{split_num}.pth'))

    print(len(train_data))
    print(len(val_data))
    print(len(test_data))

    return train_data, val_data, test_data

class Autoencoder(nn.Module):
    def __init__(self, input_dim, first_layer_dim, second_layer_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, first_layer_dim)
        self.fc2 = nn.Linear(first_layer_dim, second_layer_dim)
        self.fc3 = nn.Linear(second_layer_dim, latent_dim)
        self.fc4 = nn.Linear(latent_dim, second_layer_dim)
        self.fc5 = nn.Linear(second_layer_dim, first_layer_dim)
        self.fc6 = nn.Linear(first_layer_dim, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return self.fc3(h2)

    def decode(self, z):
        h3 = torch.relu(self.fc4(z))
        h4 = torch.relu(self.fc5(h3))
        return self.fc6(h4)

    def forward(self, x):
        z = self.encode(x)
        recon_x = self.decode(z)
        return recon_x

def ae_loss_function(recon_x, x, data_name, recon_weight=1.0):
    recon_loss = recon_weight * nn.functional.mse_loss(recon_x, x, reduction='sum')
    return recon_loss

def save_weights_to_pickle(model, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    weights = {name: param.to('cpu').detach().numpy() for name, param in model.named_parameters()}
    with open(file_name, 'wb') as handle:
        pickle.dump(weights, handle)
    print(f"Model weights saved to {file_name}")

if __name__ == '__main__':
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_path = ""  # Adjust this path if needed

    # omics = ["cna", "exp", "meth"]
    omics = ["mut"]

    for omic in omics:
        print("Omic : ",omic)
        for split_num in range(1, 6):
            run = wandb.init(project="Self-Supervised-Machine-Learning-For-Predicting-Cancer-Dependencies-Splits", entity="kemal-bayik", name=f"TCGA_{omic}_{current_time}_Split_{split_num}_Original")

            config = wandb.config
            config.learning_rate = 1e-4
            config.batch_size = 500
            config.epochs = 100
            config.patience = 10
            config.first_layer_dim = 500
            config.second_layer_dim = 200
            config.latent_dim = 50

            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

            train_data, val_data, test_data = load_split(split_num, omic, base_path)

            config.input_dim = train_data.dataset.tensors[0].shape[1]

            train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)
            test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

            model = Autoencoder(input_dim=config.input_dim, first_layer_dim=config.first_layer_dim, second_layer_dim=config.second_layer_dim, latent_dim=config.latent_dim)
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
                    recon_batch = model(inputs)
                    loss = ae_loss_function(recon_batch, inputs, omic)
                    loss.backward()
                    train_loss += loss.item()
                    optimizer.step()
                train_loss /= len(train_loader.dataset)

                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for data in val_loader:
                        inputs = data[0].to(device)
                        recon_batch = model(inputs)
                        loss = ae_loss_function(recon_batch, inputs, omic)
                        val_loss += loss.item()
                val_loss /= len(val_loader.dataset)

                wandb.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": config.learning_rate,
                    "batch_size": config.batch_size,
                    "epoch": epoch + 1,
                })

                print(f'Epoch [{epoch + 1}/{config.epochs}], Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}')

                # Early stopping
                if val_loss < best_loss:
                    best_loss = val_loss
                    early_stop_counter = 0
                    # Save the model's best weights
                    save_weights_to_pickle(model, f'PytorchStaticSplits/OriginalCode/Results/Split{split_num}/USL_Pretrained/tcga_{omic}_ae_best_split_{split_num}.pickle')

            print('\nAE training completed in %.1f mins' % ((time.time() - start_time) / 60))

            # Evaluate the model on the test set
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for data in test_loader:
                    inputs = data[0].to(device)
                    recon_batch = model(inputs)
                    loss = ae_loss_function(recon_batch, inputs, omic)
                    test_loss += loss.item()
            test_loss /= len(test_loader.dataset)

            wandb.log({
                "test_loss": test_loss
            })

            print(f'\nTest Loss: {test_loss:.6f}')

            run.finish()
