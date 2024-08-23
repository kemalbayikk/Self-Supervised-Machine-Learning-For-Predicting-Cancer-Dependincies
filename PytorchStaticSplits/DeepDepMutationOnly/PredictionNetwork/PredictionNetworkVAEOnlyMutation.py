import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm
import wandb
from datetime import datetime

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# device = "cuda"
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

class DeepDEP(nn.Module):
    def __init__(self, premodel_mut, premodel_exp, premodel_cna, premodel_meth, premodel_fprint, dense_layer_dim):
        super(DeepDEP, self).__init__()
        self.vae_mut = premodel_mut
        self.vae_fprint = premodel_fprint

        self.fc_merged1 = nn.Linear(dense_layer_dim, dense_layer_dim)
        self.fc_merged2 = nn.Linear(dense_layer_dim, dense_layer_dim)
        self.fc_out = nn.Linear(dense_layer_dim, 1)

    def forward(self, mut, exp, cna, meth, fprint):
        recon_gene, mu_fprint, logvar_gene = self.vae_fprint(fprint)
        
        merged = torch.cat([mu_fprint], dim=1)
        merged = torch.relu(self.fc_merged1(merged))
        merged = torch.relu(self.fc_merged2(merged))
        output = self.fc_out(merged)
        return output

def load_pretrained_vae(filepath, input_dim, first_layer_dim, second_layer_dim, latent_dim):
    vae = VariationalAutoencoder(input_dim, first_layer_dim, second_layer_dim, latent_dim)
    vae_state = pickle.load(open(filepath, 'rb'))

    # Convert numpy arrays to PyTorch tensors
    for key in vae_state:
        if isinstance(vae_state[key], np.ndarray):
            vae_state[key] = torch.tensor(vae_state[key])

    vae.load_state_dict(vae_state)
    return vae

def train_model(model, train_loader, test_loader, num_epoch, patience, learning_rate, split_num):
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
            "val_loss": test_loss,
            "val_pearson_correlation": pearson_corr,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "batch_size": train_loader.batch_size,
            "epoch": epoch + 1
        })

        if test_loss < best_loss:
            best_loss = test_loss
            epochs_no_improve = 0
            best_model_state_dict = model.state_dict()
            torch.save(best_model_state_dict, f'PytorchStaticSplits/DeepDepFingerprintOnly/Results/Split{split_num}/PredictionNetworkModels/VAE_Prediction_Network_Split_{split_num}_Only_Fingerprint.pth')
            print("Model saved")

    return best_model_state_dict, training_predictions, training_targets_list

if __name__ == '__main__':

    lr = 1e-3
        
    split_num = 2
 
    ccl_size = "278"
    current_time = datetime.now().strftime("%m-%d_%H-%M")
    # with open('Data/ccl_complete_data_278CCL_1298DepOI_360844samples.pickle', 'rb') as f:
    #     data_mut, data_exp, data_cna, data_meth, data_dep, data_fprint = pickle.load(f)

    with open(f'Data/data_split_{split_num}.pickle', 'rb') as f:
        train_dataset, val_dataset, test_dataset = pickle.load(f)

    run = wandb.init(project="DeepDepVAELRTestPredictionNetwork", entity="kemal-bayik", name=f"Prediction_Network_{current_time}_VAE_Split_{split_num}_Only_Fingerprint")

    config = wandb.config
    config.learning_rate = lr
    config.batch_size = 10000
    config.epochs = 100
    config.patience = 3

    latent_dim = 50

    # Define dimensions for the pretrained VAEs
    dims_mut = (train_dataset[:][0].shape[1], 1000, 100, 50)
    dims_exp = (train_dataset[:][1].shape[1], 500, 200, 50)
    dims_cna = (train_dataset[:][2].shape[1], 500, 200, 50)
    dims_meth = (train_dataset[:][3].shape[1], 500, 200, 50)
    dims_fprint = (train_dataset[:][4].shape[1], 1000, 100, 50)

    # Load pre-trained VAE models    
    premodel_mut = load_pretrained_vae(f'PytorchStaticSplits/DeepDepVAE/Results/Split{split_num}/CCL_Pretrained/ccl_mut_vae_best_split_{split_num}_Beta1_LR_Test_After.pickle', *dims_mut)
    premodel_exp = load_pretrained_vae(f'PytorchStaticSplits/DeepDepVAE/Results/Split{split_num}/CCL_Pretrained/ccl_exp_vae_best_split_{split_num}_Beta1_LR_Test_After.pickle', *dims_exp)
    premodel_cna = load_pretrained_vae(f'PytorchStaticSplits/DeepDepVAE/Results/Split{split_num}/CCL_Pretrained/ccl_cna_vae_best_split_{split_num}_Beta1_LR_Test_After.pickle', *dims_cna)
    premodel_meth = load_pretrained_vae(f'PytorchStaticSplits/DeepDepVAE/Results/Split{split_num}/CCL_Pretrained/ccl_meth_vae_best_split_{split_num}_Beta1_LR_Test_After.pickle', *dims_meth)
    premodel_fprint = load_pretrained_vae(f'PytorchStaticSplits/DeepDepVAE/Results/Split{split_num}/CCL_Pretrained/ccl_fprint_vae_best_split_{split_num}_Beta1_LR_Test_After.pickle', *dims_fprint)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Create the DeepDEP model using the pretrained VAE models
    model = DeepDEP(premodel_mut, premodel_exp, premodel_cna, premodel_meth, premodel_fprint, 50)
    best_model_state_dict, training_predictions, training_targets_list = train_model(model, train_loader, val_loader, config.epochs, config.patience, config.learning_rate, split_num)

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
    print(f"Test Pearson Correlation: {pearson_corr}")

    wandb.log({
        "test_pearson_correlation": pearson_corr,
    })
    
    # # Plot results
    y_true_train = np.array(training_targets_list).flatten()
    y_pred_train = np.array(training_predictions).flatten()
    y_true_test = np.array(targets_list).flatten()
    y_pred_test = np.array(predictions).flatten()

    np.savetxt(f'PytorchStaticSplits/DeepDepFingerprintOnly/Results/Split{split_num}/predictions/y_true_train_Prediction_Network_VAE_Split_{split_num}_Only_Fingerprint.txt', y_true_train, fmt='%.6f')
    np.savetxt(f'PytorchStaticSplits/DeepDepFingerprintOnly/Results/Split{split_num}/predictions/y_pred_train_Prediction_Network_VAE_Split_{split_num}_Only_Fingerprint.txt', y_pred_train, fmt='%.6f')
    np.savetxt(f'PytorchStaticSplits/DeepDepFingerprintOnly/Results/Split{split_num}/predictions/y_true_test_Prediction_Network_VAE_Split_{split_num}_Only_Fingerprint.txt', y_true_test, fmt='%.6f')
    np.savetxt(f'PytorchStaticSplits/DeepDepFingerprintOnly/Results/Split{split_num}/predictions/y_pred_test_Prediction_Network_VAE_Split_{split_num}_Only_Fingerprint.txt', y_pred_test, fmt='%.6f')

    print(f"Training: y_true_train size: {len(y_true_train)}, y_pred_train size: {len(y_pred_train)}")
    print(f"Testing: y_true_test size: {len(y_true_test)}, y_pred_test size: {len(y_pred_test)}")

    #plot_results(y_true_train, y_pred_train, y_true_test, y_pred_test, config.batch_size, config.learning_rate, config.epochs)
    #plot_density(y_true_train, y_pred_train, y_true_test, y_pred_test, config.batch_size, config.learning_rate, config.epochs)

    run.finish()
