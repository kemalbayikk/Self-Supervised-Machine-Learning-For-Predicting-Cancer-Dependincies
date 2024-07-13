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

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# device = "cuda"
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

def load_pretrained_lvae(filepath, input_dim, first_layer_dim, second_layer_dim, latent_dim, ladder_dim):
    lvae = LadderVariationalAutoencoder(input_dim, first_layer_dim, second_layer_dim, latent_dim, ladder_dim)
    lvae_state = pickle.load(open(filepath, 'rb'))

    # Convert numpy arrays to PyTorch tensors
    for key in lvae_state:
        if isinstance(lvae_state[key], np.ndarray):
            lvae_state[key] = torch.tensor(lvae_state[key])

    lvae.load_state_dict(lvae_state)
    return lvae


class DeepDEP(nn.Module):
    def __init__(self, premodel_mut, premodel_exp, premodel_cna, premodel_meth, premodel_fprint, dense_layer_dim):
        super(DeepDEP, self).__init__()
        self.vae_mut = premodel_mut
        self.vae_exp = premodel_exp
        self.vae_cna = premodel_cna
        self.vae_meth = premodel_meth
        self.vae_fprint = premodel_fprint

        self.fc_merged1 = nn.Linear(250, dense_layer_dim)
        self.fc_merged2 = nn.Linear(dense_layer_dim, dense_layer_dim)
        self.fc_out = nn.Linear(dense_layer_dim, 1)

    def forward(self, mut, exp, cna, meth, fprint):
        recon_mut, mu_mut, logvar_mut = self.vae_mut(mut)
        recon_exp, mu_exp, logvar_exp = self.vae_exp(exp)
        recon_cna, mu_cna, logvar_cna = self.vae_cna(cna)
        recon_meth, mu_meth, logvar_meth = self.vae_meth(meth)
        recon_gene, mu_fprint, logvar_gene = self.vae_fprint(fprint)
        
        merged = torch.cat([mu_mut, mu_exp, mu_cna, mu_meth, mu_fprint], dim=1)
        merged = torch.relu(self.fc_merged1(merged))
        merged = torch.relu(self.fc_merged2(merged))
        output = self.fc_out(merged)
        return output


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
            torch.save(best_model_state_dict, 'best_model.pth')
            print("Model saved")

    return best_model_state_dict, training_predictions, training_targets_list

if __name__ == '__main__':
    ccl_size = "278"
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # with open('Data/ccl_complete_data_28CCL_1298DepOI_36344samples_demo.pickle', 'rb') as f:
    #     data_mut, data_exp, data_cna, data_meth, data_dep, data_fprint = pickle.load(f)

    with open('Data/ccl_complete_data_278CCL_1298DepOI_360844samples.pickle', 'rb') as f:
        data_mut, data_exp, data_cna, data_meth, data_dep, data_fprint = pickle.load(f)

    wandb.init(project="Self-Supervised-Machine-Learning-For-Predicting-Cancer-Dependencies", entity="kemal-bayik", name=f"Just_NN_{ccl_size}CCL_{current_time}")

    config = wandb.config
    config.learning_rate = 1e-4
    config.batch_size = 10000
    config.epochs = 100
    config.patience = 3

    # Define dimensions for the pretrained LVAEs
    dims_mut = (data_mut.shape[1], 1000, 100, 50, 50)
    dims_exp = (data_exp.shape[1], 500, 200, 50, 50)
    dims_cna = (data_cna.shape[1], 500, 200, 50, 50)
    dims_meth = (data_meth.shape[1], 500, 200, 50, 50)
    dims_fprint = (data_fprint.shape[1], 1000, 100, 50, 50)

    # Load pre-trained LVAE models    
    premodel_mut = load_pretrained_lvae('results/ladder_variational_autoencoders/premodel_ccl_mut_lvae.pickle', *dims_mut)
    premodel_exp = load_pretrained_lvae('results/ladder_variational_autoencoders/premodel_ccl_exp_lvae.pickle', *dims_exp)
    premodel_cna = load_pretrained_lvae('results/ladder_variational_autoencoders/premodel_ccl_cna_lvae.pickle', *dims_cna)
    premodel_meth = load_pretrained_lvae('results/ladder_variational_autoencoders/premodel_ccl_meth_lvae.pickle', *dims_meth)
    premodel_fprint = load_pretrained_lvae('results/ladder_variational_autoencoders/premodel_ccl_fprint_lvae.pickle', *dims_fprint)

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

    # Create the DeepDEP model using the pretrained LVAE models
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
    torch.save(best_model_state_dict, 'results/models/deepdep_vae_model_9July.pth')
    
    # Plot results
    y_true_train = np.array(training_targets_list).flatten()
    y_pred_train = np.array(training_predictions).flatten()
    y_true_test = np.array(targets_list).flatten()
    y_pred_test = np.array(predictions).flatten()

    np.savetxt(f'results/predictions/y_true_train_CCL_LVAE.txt', y_true_train, fmt='%.6f')
    np.savetxt(f'results/predictions/y_pred_train_CCL_LVAE.txt', y_pred_train, fmt='%.6f')
    np.savetxt(f'results/predictions/y_true_test_CCL_LVAE.txt', y_true_test, fmt='%.6f')
    np.savetxt(f'results/predictions/y_pred_test_CCL_LVAE.txt', y_pred_test, fmt='%.6f')

    print(f"Training: y_true_train size: {len(y_true_train)}, y_pred_train size: {len(y_pred_train)}")
    print(f"Testing: y_true_test size: {len(y_true_test)}, y_pred_test size: {len(y_pred_test)}")
