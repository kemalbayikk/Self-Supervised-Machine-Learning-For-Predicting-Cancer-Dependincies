import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pickle
import time
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3_mean = nn.Linear(hidden_dims[1], latent_dim)
        self.fc3_log_var = nn.Linear(hidden_dims[1], latent_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return self.fc3_mean(h), self.fc3_log_var(h)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return z, mean, log_var

class DeepDEP(nn.Module):
    def __init__(self, dims_mut, dims_exp, dims_cna, dims_meth, fprint_dim, dense_layer_dim):
        super(DeepDEP, self).__init__()
        self.vae_mut = VAE(dims_mut[0], [1000, 100], 50)
        self.vae_exp = VAE(dims_exp[0], [500, 200], 50)
        self.vae_cna = VAE(dims_cna[0], [500, 200], 50)
        self.vae_meth = VAE(dims_meth[0], [500, 200], 50)

        print(fprint_dim)
        self.vae_gene = VAE(fprint_dim, [1000, 100], 50)

        # self.fc_gene1 = nn.Linear(fprint_dim, 1000)
        # self.fc_gene2 = nn.Linear(1000, 100)
        # self.fc_gene3 = nn.Linear(100, 50)

        self.fc_merged1 = nn.Linear(250, dense_layer_dim)
        self.fc_merged2 = nn.Linear(dense_layer_dim, dense_layer_dim)
        self.fc_out = nn.Linear(dense_layer_dim, 1)

    def forward(self, mut, exp, cna, meth, fprint):
        z_mut, _, _ = self.vae_mut(mut)
        z_exp, _, _ = self.vae_exp(exp)
        z_cna, _, _ = self.vae_cna(cna)
        z_meth, _, _ = self.vae_meth(meth)
        z_gene, _, _ = self.vae_gene(fprint)
        
        # gene = torch.relu(self.fc_gene1(fprint))
        # gene = torch.relu(self.fc_gene2(gene))
        # gene = torch.relu(self.fc_gene3(gene))
        
        merged = torch.cat([z_mut, z_exp, z_cna, z_meth, z_gene], dim=1)
        merged = torch.relu(self.fc_merged1(merged))
        merged = torch.relu(self.fc_merged2(merged))
        output = self.fc_out(merged)
        return output

def train_model(model, train_loader, test_loader, num_epoch, patience):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    model.to(device)

    start_time = time.time()

    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        if epoch == 0:
            print("Training started...")
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epoch}", leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            inputs = [tensor.to(device) for tensor in batch[:-1]]  # All but last element are inputs
            targets = batch[-1].to(device)  # Last element is the target

            optimizer.zero_grad()
            outputs = model(*inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

        model.eval()
        test_loss = 0.0
        predictions = []
        targets_list = []
        with torch.no_grad():
            for batch in test_loader:
                inputs = [tensor.to(device) for tensor in batch[:-1]]  # All but last element are inputs
                targets = batch[-1].to(device)  # Last element is the target
                outputs = model(*inputs)
                test_loss += criterion(outputs, targets).item()
                
                predictions.extend(outputs.cpu().numpy())
                targets_list.extend(targets.cpu().numpy())

        test_loss /= len(test_loader)
        print(f"Test Loss: {test_loss}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed in: {elapsed_time / 60:.2f} minutes")

    # Calculate Pearson Correlation Coefficient
    predictions = np.array(predictions).flatten()
    targets = np.array(targets_list).flatten()
    pearson_corr, _ = pearsonr(predictions, targets)
    print(f"Pearson Correlation: {pearson_corr}")

    return model

if __name__ == '__main__':
    # Load your data here as appropriate
    # with open('Data/ccl_complete_data_28CCL_1298DepOI_36344samples_demo.pickle', 'rb') as f:
    #     data_mut, data_exp, data_cna, data_meth, data_dep, data_fprint = pickle.load(f)

    with open('Data/ccl_complete_data_278CCL_1298DepOI_360844samples.pickle', 'rb') as f:
        data_mut, data_exp, data_cna, data_meth, data_dep, data_fprint = pickle.load(f)

    # You should add normalization or other preprocessing as needed

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

    train_loader = DataLoader(train_dataset, batch_size=500, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)

    # Assuming dimensions of inputs are known or computed from your data
    model = DeepDEP((data_mut.shape[1],), (data_exp.shape[1],), (data_cna.shape[1],), (data_meth.shape[1],), data_fprint.shape[1], 250)
    trained_model = train_model(model, train_loader, test_loader, 5, 3)

    # Save the model
    torch.save(trained_model.state_dict(), 'model_demo_vae.pth')
