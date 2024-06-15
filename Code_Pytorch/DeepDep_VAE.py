# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset, random_split
# import pickle
# import time
# from scipy.stats import pearsonr
# import numpy as np
# from tqdm import tqdm

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# #device = "cuda"
# print(device)

# class VariationalAutoencoder(nn.Module):
#     def __init__(self, input_dim, first_layer_dim, second_layer_dim, latent_dim):
#         super(VariationalAutoencoder, self).__init__()
#         self.fc1 = nn.Linear(input_dim, first_layer_dim)
#         self.fc2 = nn.Linear(first_layer_dim, second_layer_dim)
#         self.fc31 = nn.Linear(second_layer_dim, latent_dim)
#         self.fc32 = nn.Linear(second_layer_dim, latent_dim)
#         self.fc4 = nn.Linear(latent_dim, second_layer_dim)
#         self.fc5 = nn.Linear(second_layer_dim, first_layer_dim)
#         self.fc6 = nn.Linear(first_layer_dim, input_dim)

#     def encode(self, x):
#         h1 = torch.relu(self.fc1(x))
#         h2 = torch.relu(self.fc2(h1))
#         return self.fc31(h2), self.fc32(h2)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def decode(self, z):
#         h3 = torch.relu(self.fc4(z))
#         h4 = torch.relu(self.fc5(h3))
#         return self.fc6(h4)

#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         return z
# class DeepDEP(nn.Module):
#     def __init__(self, premodel_mut, premodel_exp, premodel_cna, premodel_meth, fprint_dim, dense_layer_dim):
#         super(DeepDEP, self).__init__()
#         self.vae_mut = premodel_mut
#         self.vae_exp = premodel_exp
#         self.vae_cna = premodel_cna
#         self.vae_meth = premodel_meth

#         self.vae_gene = VariationalAutoencoder(fprint_dim, 1000, 100, 50)

#         self.fc_merged1 = nn.Linear(250, dense_layer_dim)
#         self.fc_merged2 = nn.Linear(dense_layer_dim, dense_layer_dim)
#         self.fc_out = nn.Linear(dense_layer_dim, 1)

#     def forward(self, mut, exp, cna, meth, fprint):
#         z_mut = self.vae_mut(mut)
#         z_exp = self.vae_exp(exp)
#         z_cna = self.vae_cna(cna)
#         z_meth = self.vae_meth(meth)
#         z_gene = self.vae_gene(fprint)
        
#         merged = torch.cat([z_mut, z_exp, z_cna, z_meth, z_gene], dim=1)
#         merged = torch.relu(self.fc_merged1(merged))
#         merged = torch.relu(self.fc_merged2(merged))
#         output = self.fc_out(merged)
#         return output
    
# def load_pretrained_vae(filepath, input_dim, first_layer_dim, second_layer_dim, latent_dim):
#     vae = VariationalAutoencoder(input_dim, first_layer_dim, second_layer_dim, latent_dim)
#     vae_state = pickle.load(open(filepath, 'rb'))
    
#     # Convert numpy arrays to PyTorch tensors
#     for key in vae_state:
#         vae_state[key] = torch.tensor(vae_state[key])
    
#     vae.load_state_dict(vae_state)
#     return vae

# def train_model(model, train_loader, test_loader, num_epoch):
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters())
#     model.to(device)

#     start_time = time.time()

#     for epoch in range(num_epoch):
#         model.train()
#         running_loss = 0.0
#         if epoch == 0:
#             print("Training started...")
#         progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epoch}", leave=False)
#         for batch_idx, batch in enumerate(progress_bar):
#             inputs = [tensor.to(device) for tensor in batch[:-1]]  # All but last element are inputs
#             targets = batch[-1].to(device)  # Last element is the target

#             optimizer.zero_grad()
#             outputs = model(*inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#         print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

#         model.eval()
#         test_loss = 0.0
#         predictions = []
#         targets_list = []
#         with torch.no_grad():
#             for batch in test_loader:
#                 inputs = [tensor.to(device) for tensor in batch[:-1]]  # All but last element are inputs
#                 targets = batch[-1].to(device)  # Last element is the target
#                 outputs = model(*inputs)
#                 test_loss += criterion(outputs, targets).item()
                
#                 predictions.extend(outputs.cpu().numpy())
#                 targets_list.extend(targets.cpu().numpy())

#         test_loss /= len(test_loader)
#         print(f"Test Loss: {test_loss}")

#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"Training completed in: {elapsed_time / 60:.2f} minutes")

#     # Calculate Pearson Correlation Coefficient
#     predictions = np.array(predictions).flatten()
#     targets = np.array(targets_list).flatten()
#     pearson_corr, _ = pearsonr(predictions, targets)
#     print(f"Pearson Correlation: {pearson_corr}")

#     return model

# if __name__ == '__main__':
#     # Load your data here as appropriate
#     # with open('Data/ccl_complete_data_278CCL_1298DepOI_360844samples.pickle', 'rb') as f:
#     #     data_mut, data_exp, data_cna, data_meth, data_dep, data_fprint = pickle.load(f)

#     with open('Data/ccl_complete_data_28CCL_1298DepOI_36344samples_demo.pickle', 'rb') as f:
#         data_mut, data_exp, data_cna, data_meth, data_dep, data_fprint = pickle.load(f)

#     # Define dimensions for the pretrained VAEs
#     dims_mut = (data_mut.shape[1], 1000, 100, 50)
#     dims_exp = (data_exp.shape[1], 500, 200, 50)
#     dims_cna = (data_cna.shape[1], 500, 200, 50)
#     dims_meth = (data_meth.shape[1], 500, 200, 50)

#     # Load pre-trained VAE models
#     premodel_mut = load_pretrained_vae('../results/variational_autoencoders/premodel_tcga_mut_vae_1000_100_50.pickle', *dims_mut)
#     premodel_exp = load_pretrained_vae('../results/variational_autoencoders/premodel_tcga_exp_vae_500_200_50.pickle', *dims_exp)
#     premodel_cna = load_pretrained_vae('../results/variational_autoencoders/premodel_tcga_cna_vae_500_200_50.pickle', *dims_cna)
#     premodel_meth = load_pretrained_vae('../results/variational_autoencoders/premodel_tcga_meth_vae_500_200_50.pickle', *dims_meth)

#     # Convert numpy arrays to PyTorch tensors and create datasets
#     tensor_mut = torch.Tensor(data_mut)
#     tensor_exp = torch.Tensor(data_exp)
#     tensor_cna = torch.Tensor(data_cna)
#     tensor_meth = torch.Tensor(data_meth)
#     tensor_dep = torch.Tensor(data_dep)
#     tensor_fprint = torch.Tensor(data_fprint)

#     dataset = TensorDataset(tensor_mut, tensor_exp, tensor_cna, tensor_meth, tensor_fprint, tensor_dep)

#     # Train/test split
#     train_size = int(0.8 * len(dataset))
#     test_size = len(dataset) - train_size
#     train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

#     train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)

#     # Create the DeepDEP model using the pretrained VAE models
#     model = DeepDEP(premodel_mut, premodel_exp, premodel_cna, premodel_meth, data_fprint.shape[1], 250)
#     trained_model = train_model(model, train_loader, test_loader, 100)

#     # Save the model
#     torch.save(trained_model.state_dict(), 'model_demo_vae.pth')

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset, random_split
# import pickle
# import time
# from scipy.stats import pearsonr
# import numpy as np
# from tqdm import tqdm

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# #device = "cuda"
# print(device)

# class VariationalAutoencoder(nn.Module):
#     def __init__(self, input_dim, first_layer_dim, second_layer_dim, latent_dim):
#         super(VariationalAutoencoder, self).__init__()
#         self.fc1 = nn.Linear(input_dim, first_layer_dim)
#         self.fc2 = nn.Linear(first_layer_dim, second_layer_dim)
#         self.fc31 = nn.Linear(second_layer_dim, latent_dim)
#         self.fc32 = nn.Linear(second_layer_dim, latent_dim)
#         self.fc4 = nn.Linear(latent_dim, second_layer_dim)
#         self.fc5 = nn.Linear(second_layer_dim, first_layer_dim)
#         self.fc6 = nn.Linear(first_layer_dim, input_dim)

#     def encode(self, x):
#         h1 = torch.relu(self.fc1(x))
#         h2 = torch.relu(self.fc2(h1))
#         return self.fc31(h2), self.fc32(h2)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def decode(self, z):
#         h3 = torch.relu(self.fc4(z))
#         h4 = torch.relu(self.fc5(h3))
#         return self.fc6(h4)

#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         recon_x = self.decode(z)
#         return recon_x, mu, logvar

# class DeepDEP(nn.Module):
#     def __init__(self, premodel_mut, premodel_exp, premodel_cna, premodel_meth, fprint_dim, dense_layer_dim):
#         super(DeepDEP, self).__init__()
#         self.vae_mut = premodel_mut
#         self.vae_exp = premodel_exp
#         self.vae_cna = premodel_cna
#         self.vae_meth = premodel_meth

#         self.vae_gene = VariationalAutoencoder(fprint_dim, 1000, 100, 50)

#         self.fc_merged1 = nn.Linear(250, dense_layer_dim)
#         self.fc_merged2 = nn.Linear(dense_layer_dim, dense_layer_dim)
#         self.fc_out = nn.Linear(dense_layer_dim, 1)

#     def forward(self, mut, exp, cna, meth, fprint):
#         recon_mut, mu_mut, logvar_mut = self.vae_mut(mut)
#         recon_exp, mu_exp, logvar_exp = self.vae_exp(exp)
#         recon_cna, mu_cna, logvar_cna = self.vae_cna(cna)
#         recon_meth, mu_meth, logvar_meth = self.vae_meth(meth)
#         recon_gene, mu_gene, logvar_gene = self.vae_gene(fprint)
        
#         merged = torch.cat([mu_mut, mu_exp, mu_cna, mu_meth, mu_gene], dim=1)
#         merged = torch.relu(self.fc_merged1(merged))
#         merged = torch.relu(self.fc_merged2(merged))
#         output = self.fc_out(merged)
#         return output, recon_mut, mut, mu_mut, logvar_mut, recon_exp, exp, mu_exp, logvar_exp, recon_cna, cna, mu_cna, logvar_cna, recon_meth, meth, mu_meth, logvar_meth, recon_gene, fprint, mu_gene, logvar_gene

# def vae_loss_function(recon_x, x, mu, logvar):
#     recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
#     kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     return recon_loss + kl_loss

# def total_loss_function(outputs, targets):
#     output, recon_mut, mut, mu_mut, logvar_mut, recon_exp, exp, mu_exp, logvar_exp, recon_cna, cna, mu_cna, logvar_cna, recon_meth, meth, mu_meth, logvar_meth, recon_gene, fprint, mu_gene, logvar_gene = outputs

#     recon_loss_mut = vae_loss_function(recon_mut, mut, mu_mut, logvar_mut)
#     recon_loss_exp = vae_loss_function(recon_exp, exp, mu_exp, logvar_exp)
#     recon_loss_cna = vae_loss_function(recon_cna, cna, mu_cna, logvar_cna)
#     recon_loss_meth = vae_loss_function(recon_meth, meth, mu_meth, logvar_meth)
#     recon_loss_gene = vae_loss_function(recon_gene, fprint, mu_gene, logvar_gene)

#     return nn.functional.mse_loss(output, targets) + recon_loss_mut + recon_loss_exp + recon_loss_cna + recon_loss_meth + recon_loss_gene

# def load_pretrained_vae(filepath, input_dim, first_layer_dim, second_layer_dim, latent_dim):
#     vae = VariationalAutoencoder(input_dim, first_layer_dim, second_layer_dim, latent_dim)
#     vae_state = pickle.load(open(filepath, 'rb'))
    
#     # Convert numpy arrays to PyTorch tensors
#     for key in vae_state:
#         if isinstance(vae_state[key], np.ndarray):
#             vae_state[key] = torch.tensor(vae_state[key])
    
#     vae.load_state_dict(vae_state)
#     return vae

# def train_model(model, train_loader, test_loader, num_epoch, patience):
#     optimizer = optim.Adam(model.parameters())
#     model.to(device)

#     start_time = time.time()

#     best_loss = float('inf')
#     epochs_no_improve = 0
#     early_stop = False

#     for epoch in range(num_epoch):
#         if early_stop:
#             break

#         model.train()
#         running_loss = 0.0
#         if epoch == 0:
#             print("Training started...")
#         progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epoch}", leave=False)
#         for batch_idx, batch in enumerate(progress_bar):
#             inputs = [tensor.to(device) for tensor in batch[:-1]]
#             targets = batch[-1].to(device)

#             optimizer.zero_grad()
#             outputs = model(*inputs)
#             loss = total_loss_function(outputs, targets)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#         print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

#         model.eval()
#         test_loss = 0.0
#         predictions = []
#         targets_list = []
#         with torch.no_grad():
#             for batch in test_loader:
#                 inputs = [tensor.to(device) for tensor in batch[:-1]]
#                 targets = batch[-1].to(device)
#                 outputs = model(*inputs)
#                 loss = total_loss_function(outputs, targets)
#                 test_loss += loss.item()
                
#                 predictions.extend(outputs[0].cpu().numpy())
#                 targets_list.extend(targets.cpu().numpy())

#         test_loss /= len(test_loader)
#         print(f"Test Loss: {test_loss}")

#         predictions = np.array(predictions).flatten()
#         targets = np.array(targets_list).flatten()
#         pearson_corr, _ = pearsonr(predictions, targets)
#         print(f"Pearson Correlation: {pearson_corr}")

#         if test_loss < best_loss:
#             best_loss = test_loss
#             epochs_no_improve = 0
#             torch.save(model.state_dict(), 'best_model.pth')
#         else:
#             epochs_no_improve += 1
#             if epochs_no_improve >= patience:
#                 print("Early stopping")
#                 early_stop = True

#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"Training completed in: {elapsed_time / 60:.2f} minutes")

#     predictions = np.array(predictions).flatten()
#     targets = np.array(targets_list).flatten()
#     pearson_corr, _ = pearsonr(predictions, targets)
#     print(f"Pearson Correlation: {pearson_corr}")

#     return model

# if __name__ == '__main__':
#     # Load your data here as appropriate
#     # with open('Data/ccl_complete_data_278CCL_1298DepOI_360844samples.pickle', 'rb') as f:
#     #     data_mut, data_exp, data_cna, data_meth, data_dep, data_fprint = pickle.load(f)

#     with open('Data/ccl_complete_data_28CCL_1298DepOI_36344samples_demo.pickle', 'rb') as f:
#         data_mut, data_exp, data_cna, data_meth, data_dep, data_fprint = pickle.load(f)

#     # Define dimensions for the pretrained VAEs
#     dims_mut = (data_mut.shape[1], 1000, 100, 50)
#     dims_exp = (data_exp.shape[1], 500, 200, 50)
#     dims_cna = (data_cna.shape[1], 500, 200, 50)
#     dims_meth = (data_meth.shape[1], 500, 200, 50)

#     # Load pre-trained VAE models
#     premodel_mut = load_pretrained_vae('../results/variational_autoencoders/premodel_tcga_mut_vae_1000_100_50.pickle', *dims_mut)
#     premodel_exp = load_pretrained_vae('../results/variational_autoencoders/premodel_tcga_exp_vae_500_200_50.pickle', *dims_exp)
#     premodel_cna = load_pretrained_vae('../results/variational_autoencoders/premodel_tcga_cna_vae_500_200_50.pickle', *dims_cna)
#     premodel_meth = load_pretrained_vae('../results/variational_autoencoders/premodel_tcga_meth_vae_500_200_50.pickle', *dims_meth)

#     # Convert numpy arrays to PyTorch tensors and create datasets
#     tensor_mut = torch.Tensor(data_mut)
#     tensor_exp = torch.Tensor(data_exp)
#     tensor_cna = torch.Tensor(data_cna)
#     tensor_meth = torch.Tensor(data_meth)
#     tensor_dep = torch.Tensor(data_dep)
#     tensor_fprint = torch.Tensor(data_fprint)

#     dataset = TensorDataset(tensor_mut, tensor_exp, tensor_cna, tensor_meth, tensor_fprint, tensor_dep)

#     # Train/test split
#     train_size = int(0.8 * len(dataset))
#     test_size = len(dataset) - train_size
#     train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#     # Create the DeepDEP model using the pretrained VAE models
#     model = DeepDEP(premodel_mut, premodel_exp, premodel_cna, premodel_meth, data_fprint.shape[1], 250)
#     trained_model = train_model(model, train_loader, test_loader, 100, 10)

#     # Save the model
#     torch.save(trained_model.state_dict(), 'model_demo_vae.pth')



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pickle
import time
from scipy.stats import pearsonr
import numpy as np
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
        recon_x = self.decode(z)
        return recon_x, mu, logvar

class DeepDEP(nn.Module):
    def __init__(self, premodel_mut, premodel_exp, premodel_cna, premodel_meth, fprint_dim, dense_layer_dim):
        super(DeepDEP, self).__init__()
        self.vae_mut = premodel_mut
        self.vae_exp = premodel_exp
        self.vae_cna = premodel_cna
        self.vae_meth = premodel_meth

        self.vae_gene = VariationalAutoencoder(fprint_dim, 1000, 100, 50)

        self.fc_merged1 = nn.Linear(250, dense_layer_dim)
        self.fc_merged2 = nn.Linear(dense_layer_dim, dense_layer_dim)
        self.fc_out = nn.Linear(dense_layer_dim, 1)

    def forward(self, mut, exp, cna, meth, fprint):
        recon_mut, mu_mut, logvar_mut = self.vae_mut(mut)
        recon_exp, mu_exp, logvar_exp = self.vae_exp(exp)
        recon_cna, mu_cna, logvar_cna = self.vae_cna(cna)
        recon_meth, mu_meth, logvar_meth = self.vae_meth(meth)
        recon_gene, mu_gene, logvar_gene = self.vae_gene(fprint)
        
        merged = torch.cat([mu_mut, mu_exp, mu_cna, mu_meth, mu_gene], dim=1)
        merged = torch.relu(self.fc_merged1(merged))
        merged = torch.relu(self.fc_merged2(merged))
        output = self.fc_out(merged)
        return output, recon_mut, mut, mu_mut, logvar_mut, recon_exp, exp, mu_exp, logvar_exp, recon_cna, cna, mu_cna, logvar_cna, recon_meth, meth, mu_meth, logvar_meth, recon_gene, fprint, mu_gene, logvar_gene

def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

def total_loss_function(outputs, targets):
    output, recon_mut, mut, mu_mut, logvar_mut, recon_exp, exp, mu_exp, logvar_exp, recon_cna, cna, mu_cna, logvar_cna, recon_meth, meth, mu_meth, logvar_meth, recon_gene, fprint, mu_gene, logvar_gene = outputs

    recon_loss_mut = vae_loss_function(recon_mut, mut, mu_mut, logvar_mut)
    recon_loss_exp = vae_loss_function(recon_exp, exp, mu_exp, logvar_exp)
    recon_loss_cna = vae_loss_function(recon_cna, cna, mu_cna, logvar_cna)
    recon_loss_meth = vae_loss_function(recon_meth, meth, mu_meth, logvar_meth)
    recon_loss_gene = vae_loss_function(recon_gene, fprint, mu_gene, logvar_gene)

    return nn.functional.mse_loss(output, targets) + recon_loss_mut + recon_loss_exp + recon_loss_cna + recon_loss_meth + recon_loss_gene

def load_pretrained_vae(filepath, input_dim, first_layer_dim, second_layer_dim, latent_dim):
    vae = VariationalAutoencoder(input_dim, first_layer_dim, second_layer_dim, latent_dim)
    vae_state = pickle.load(open(filepath, 'rb'))
    
    # Convert numpy arrays to PyTorch tensors
    for key in vae_state:
        if isinstance(vae_state[key], np.ndarray):
            vae_state[key] = torch.tensor(vae_state[key])
    
    vae.load_state_dict(vae_state)
    return vae

def train_model(model, train_loader, test_loader, num_epoch, patience, learning_rate):
    optimizer = optim.Adam(model.parameters())
    model.to(device)

    start_time = time.time()

    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epoch):
        if early_stop:
            break

        model.train()
        running_loss = 0.0
        if epoch == 0:
            print("Training started...")
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epoch}", leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            inputs = [tensor.to(device) for tensor in batch[:-1]]
            targets = batch[-1].to(device)

            optimizer.zero_grad()
            outputs = model(*inputs)
            loss = total_loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {train_loss}")

        model.eval()
        test_loss = 0.0
        predictions = []
        targets_list = []
        with torch.no_grad():
            for batch in test_loader:
                inputs = [tensor.to(device) for tensor in batch[:-1]]
                targets = batch[-1].to(device)
                outputs = model(*inputs)
                loss = total_loss_function(outputs, targets)
                test_loss += loss.item()
                
                predictions.extend(outputs[0].cpu().numpy())
                targets_list.extend(targets.cpu().numpy())

        test_loss /= len(test_loader)
        print(f"Test Loss: {test_loss}")

        predictions = np.array(predictions).flatten()
        targets = np.array(targets_list).flatten()
        pearson_corr, _ = pearsonr(predictions, targets)
        print(f"Pearson Correlation: {pearson_corr}")

        wandb.log({
            "train_loss": train_loss,
            "test_loss": test_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "batch_size": train_loader.batch_size,
            "epoch": epoch + 1,
            "pearson_correlation": pearson_corr
        })

        # En iyi modeli kaydetme
        if test_loss < best_loss:
            best_loss = test_loss
            epochs_no_improve = 0
            best_model_state_dict = model.state_dict()
            torch.save(best_model_state_dict, 'best_model.pth')
            print("Model saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping")
                early_stop = True

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed in: {elapsed_time / 60:.2f} minutes")

    return best_model_state_dict  # En iyi modelin state_dict'ini döndür

if __name__ == '__main__':

    ccl_size = "28"
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Load your data here as appropriate
    # with open('Data/ccl_complete_data_278CCL_1298DepOI_360844samples.pickle', 'rb') as f:
    #     data_mut, data_exp, data_cna, data_meth, data_dep, data_fprint = pickle.load(f)

    with open('Data/ccl_complete_data_28CCL_1298DepOI_36344samples_demo.pickle', 'rb') as f:
        data_mut, data_exp, data_cna, data_meth, data_dep, data_fprint = pickle.load(f)

    wandb.init(project="Self-Supervised-Machine-Learning-For-Predicting-Cancer-Dependincies", entity="kemal-bayik", name=f"DeepDEP_{ccl_size}CCL_{current_time}")

    config = wandb.config
    config.learning_rate = 1e-4
    config.batch_size = 256
    config.epochs = 100
    config.patience = 5

    # Define dimensions for the pretrained VAEs
    dims_mut = (data_mut.shape[1], 1000, 100, 50)
    dims_exp = (data_exp.shape[1], 500, 200, 50)
    dims_cna = (data_cna.shape[1], 500, 200, 50)
    dims_meth = (data_meth.shape[1], 500, 200, 50)

    # Load pre-trained VAE models
    premodel_mut = load_pretrained_vae('../results/variational_autoencoders/premodel_tcga_mut_vae_1000_100_50.pickle', *dims_mut)
    premodel_exp = load_pretrained_vae('../results/variational_autoencoders/premodel_tcga_exp_vae_500_200_50.pickle', *dims_exp)
    premodel_cna = load_pretrained_vae('../results/variational_autoencoders/premodel_tcga_cna_vae_500_200_50.pickle', *dims_cna)
    premodel_meth = load_pretrained_vae('../results/variational_autoencoders/premodel_tcga_meth_vae_500_200_50.pickle', *dims_meth)

    # Convert numpy arrays to PyTorch tensors and create datasets
    tensor_mut = torch.Tensor(data_mut)
    tensor_exp = torch.Tensor(data_exp)
    tensor_cna = torch.Tensor(data_cna)
    tensor_meth = torch.Tensor(data_meth)
    tensor_dep = torch.Tensor(data_dep)
    tensor_fprint = torch.Tensor(data_fprint)

    dataset = TensorDataset(tensor_mut, tensor_exp, tensor_cna, tensor_meth, tensor_fprint, tensor_dep)

    # Train/test split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Create the DeepDEP model using the pretrained VAE models
    model = DeepDEP(premodel_mut, premodel_exp, premodel_cna, premodel_meth, data_fprint.shape[1], 250)
    best_model_state_dict = train_model(model, train_loader, test_loader, config.epochs, config.patience, config.learning_rate)

    # En iyi modeli yükleyip Pearson Korelasyonunu hesaplama
    model.load_state_dict(best_model_state_dict)
    model.eval()
    predictions = []
    targets_list = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = [tensor.to(device) for tensor in batch[:-1]]
            targets = batch[-1].to(device)
            outputs = model(*inputs)
            predictions.extend(outputs[0].cpu().numpy())
            targets_list.extend(targets.cpu().numpy())

    predictions = np.array(predictions).flatten()
    targets = np.array(targets_list).flatten()
    pearson_corr, _ = pearsonr(predictions, targets)
    print(f"Best Pearson Correlation: {pearson_corr}")

    wandb.log({
        "best_pearson_correlation": pearson_corr,
    })

    # Save the best model
    torch.save(best_model_state_dict, 'model_demo_vae.pth')

