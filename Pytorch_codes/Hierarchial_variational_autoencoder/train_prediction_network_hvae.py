import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pickle
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm
import wandb
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sns

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
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

class DeepDEP(nn.Module):
    def __init__(self, premodel_mut, premodel_exp, premodel_cna, premodel_meth, premodel_fprint, dense_layer_dim):
        super(DeepDEP, self).__init__()
        self.hvae_mut = premodel_mut
        self.hvae_exp = premodel_exp
        self.hvae_cna = premodel_cna
        self.hvae_meth = premodel_meth
        self.hvae_fprint = premodel_fprint

        self.fc_merged1 = nn.Linear(125, dense_layer_dim)  # Adjusted input dimension
        self.fc_merged2 = nn.Linear(dense_layer_dim, dense_layer_dim)
        self.fc_out = nn.Linear(dense_layer_dim, 1)

    def forward(self, mut, exp, cna, meth, fprint):
        recon_mut, mu1_mut, logvar1_mut, mu2_mut, logvar2_mut = self.hvae_mut(mut)
        recon_exp, mu1_exp, logvar1_exp, mu2_exp, logvar2_exp = self.hvae_exp(exp)
        recon_cna, mu1_cna, logvar1_cna, mu2_cna, logvar2_cna = self.hvae_cna(cna)
        recon_meth, mu1_meth, logvar1_meth, mu2_meth, logvar2_meth = self.hvae_meth(meth)
        recon_gene, mu1_fprint, logvar1_fprint, mu2_fprint, logvar2_fprint = self.hvae_fprint(fprint)
        
        merged = torch.cat([mu2_mut, mu2_exp, mu2_cna, mu2_meth, mu2_fprint], dim=1)  # Using second level latent spaces
        merged = torch.relu(self.fc_merged1(merged))
        merged = torch.relu(self.fc_merged2(merged))
        output = self.fc_out(merged)
        return output

def load_pretrained_hvae(filepath, input_dim, first_layer_dim, second_layer_dim, latent_dim1, latent_dim2):
    hvae = HierarchicalVariationalAutoencoder(input_dim, first_layer_dim, second_layer_dim, latent_dim1, latent_dim2)
    hvae_state = pickle.load(open(filepath, 'rb'))
    
    # Convert numpy arrays to PyTorch tensors
    for key in hvae_state:
        if isinstance(hvae_state[key], np.ndarray):
            hvae_state[key] = torch.tensor(hvae_state[key])
    
    hvae.load_state_dict(hvae_state)
    return hvae

def train_model(model, train_loader, test_loader, num_epoch, patience, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epoch):
        if early_stop:
            break

        training_predictions = []
        training_targets_list = []

        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epoch}", leave=False)
        for batch in progress_bar:
            inputs = [tensor.to(device) for tensor in batch[:-1]]
            targets = batch[-1].to(device)

            optimizer.zero_grad()
            outputs = model(*inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            training_predictions.extend(outputs.detach().cpu().numpy())
            training_targets_list.extend(targets.detach().cpu().numpy())

        train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss}")

        model.eval()
        test_loss = 0.0
        predictions = []
        targets_list = []
        with torch.no_grad():
            for batch in test_loader:
                inputs = [tensor.to(device) for tensor in batch[:-1]]
                targets = batch[-1].to(device)
                outputs = model(*inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                
                predictions.extend(outputs.cpu().numpy())
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
            "pearson_correlation": pearson_corr,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "batch_size": train_loader.batch_size,
            "epoch": epoch + 1
        })

        if test_loss < best_loss:
            best_loss = test_loss
            epochs_no_improve = 0
            best_model_state_dict = model.state_dict()
            torch.save(best_model_state_dict, 'best_model_hvae_demo.pth')
            print("Model saved")
        # else:
        #     epochs_no_improve += 1
        #     if epochs_no_improve >= patience:
        #         print("Early stopping")
        #         early_stop = True

    return best_model_state_dict, training_predictions, training_targets_list

if __name__ == '__main__':
    ccl_size = "28"
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # with open('Data/ccl_complete_data_28CCL_1298DepOI_36344samples_demo.pickle', 'rb') as f:
    #     data_mut, data_exp, data_cna, data_meth, data_dep, data_fprint = pickle.load(f)

    with open('Data/ccl_complete_data_278CCL_1298DepOI_360844samples.pickle', 'rb') as f:
        data_mut, data_exp, data_cna, data_meth, data_dep, data_fprint = pickle.load(f)

    wandb.init(project="Self-Supervised-Machine-Learning-For-Predicting-Cancer-Dependencies", entity="kemal-bayik", name=f"Just_NN_{ccl_size}CCL_{current_time}_HVAE")

    config = wandb.config
    config.learning_rate = 1e-4
    config.batch_size = 10000
    config.epochs = 100
    config.patience = 3

    # Define dimensions for the pretrained HVAEs
    dims_mut = (data_mut.shape[1], 1000, 100, 50, 25)
    dims_exp = (data_exp.shape[1], 500, 200, 50, 25)
    dims_cna = (data_cna.shape[1], 500, 200, 50, 25)
    dims_meth = (data_meth.shape[1], 500, 200, 50, 25)
    dims_fprint = (data_fprint.shape[1], 1000, 100, 50, 25)

    # Load pre-trained HVAE models    
    premodel_mut = load_pretrained_hvae('results/hierarchial_variational_autoencoders/premodel_ccl_mut_hvae.pickle', *dims_mut)
    premodel_exp = load_pretrained_hvae('results/hierarchial_variational_autoencoders/premodel_ccl_exp_hvae.pickle', *dims_exp)
    premodel_cna = load_pretrained_hvae('results/hierarchial_variational_autoencoders/premodel_ccl_cna_hvae.pickle', *dims_cna)
    premodel_meth = load_pretrained_hvae('results/hierarchial_variational_autoencoders/premodel_ccl_meth_hvae.pickle', *dims_meth)
    premodel_fprint = load_pretrained_hvae('results/hierarchial_variational_autoencoders/premodel_ccl_fprint_hvae.pickle', *dims_fprint)

    # Convert numpy arrays to PyTorch tensors and create datasets
    tensor_mut = torch.tensor(data_mut, dtype=torch.float32)
    tensor_exp = torch.tensor(data_exp, dtype=torch.float32)
    tensor_cna = torch.tensor(data_cna, dtype=torch.float32)
    tensor_meth = torch.tensor(data_meth, dtype=torch.float32)
    tensor_dep = torch.tensor(data_dep, dtype=torch.float32)
    tensor_fprint = torch.tensor(data_fprint, dtype=torch.float32)

    dataset = TensorDataset(tensor_mut, tensor_exp, tensor_cna, tensor_meth, tensor_fprint, tensor_dep)

    # Train/test split
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print("Train size : ", train_size)
    print("Test size : ", test_size)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Create the DeepDEP model using the pretrained HVAE models
    model = DeepDEP(premodel_mut, premodel_exp, premodel_cna, premodel_meth, premodel_fprint, 250)
    best_model_state_dict, training_predictions, training_targets_list = train_model(model, train_loader, test_loader, config.epochs, config.patience, config.learning_rate)

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
            predictions.extend(outputs.cpu().numpy())
            targets_list.extend(targets.cpu().numpy())

    predictions = np.array(predictions).flatten()
    targets = np.array(targets_list).flatten()
    pearson_corr, _ = pearsonr(predictions, targets)
    print(f"Best Pearson Correlation: {pearson_corr}")

    wandb.log({
        "best_pearson_correlation": pearson_corr,
    })

    # Save the best model
    torch.save(best_model_state_dict, 'results/models/deepdep_hvae_model.pth')
    
    # Plot results
    y_true_train = np.array(training_targets_list).flatten()
    y_pred_train = np.array(training_predictions).flatten()
    y_true_test = np.array(targets_list).flatten()
    y_pred_test = np.array(predictions).flatten()

    np.savetxt(f'results/predictions/y_true_train_CCL_HVAE.txt', y_true_train, fmt='%.6f')
    np.savetxt(f'results/predictions/y_pred_train_CCL_HVAE.txt', y_pred_train, fmt='%.6f')
    np.savetxt(f'results/predictions/y_true_test_CCL_HVAE.txt', y_true_test, fmt='%.6f')
    np.savetxt(f'results/predictions/y_pred_test_CCL_HVAE.txt', y_pred_test, fmt='%.6f')

    print(f"Training: y_true_train size: {len(y_true_train)}, y_pred_train size: {len(y_pred_train)}")
    print(f"Testing: y_true_test size: {len(y_true_test)}, y_pred_test size: {len(y_pred_test)}")

    # Plotting function (if needed)
    #plot_results(y_true_train, y_pred_train, y_true_test, y_pred_test, config.batch_size, config.learning_rate, config.epochs)
    #plot_density(y_true_train, y_pred_train, y_true_test, y_pred_test, config.batch_size, config.learning_rate, config.epochs)
