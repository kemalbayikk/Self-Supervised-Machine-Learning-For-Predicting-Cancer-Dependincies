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

    def forward(self, x):
        z = self.encode(x)
        return z

class DeepDEP(nn.Module):
    def __init__(self, premodel_mut, premodel_exp, premodel_cna, premodel_meth, premodel_fprint, latent_dim, dense_layer_dim):
        super(DeepDEP, self).__init__()
        self.ae_mut = premodel_mut
        self.ae_exp = premodel_exp
        self.ae_cna = premodel_cna
        self.ae_meth = premodel_meth
        self.ae_fprint = premodel_fprint

        # Calculate the total dimension after concatenating latent dimensions
        latent_dim_total = latent_dim * 5  # 5 autoencoders with latent_dim each
        self.fc_merged1 = nn.Linear(latent_dim_total, dense_layer_dim)
        self.fc_merged2 = nn.Linear(dense_layer_dim, dense_layer_dim)
        self.fc_out = nn.Linear(dense_layer_dim, 1)

    def forward(self, mut, exp, cna, meth, fprint):
        latent_mut = self.ae_mut(mut)
        latent_exp = self.ae_exp(exp)
        latent_cna = self.ae_cna(cna)
        latent_meth = self.ae_meth(meth)
        latent_fprint = self.ae_fprint(fprint)
        
        merged = torch.cat([latent_mut, latent_exp, latent_cna, latent_meth, latent_fprint], dim=1)
        merged = torch.relu(self.fc_merged1(merged))
        merged = torch.relu(self.fc_merged2(merged))
        output = self.fc_out(merged)
        return output

def load_pretrained_ae(filepath, input_dim, first_layer_dim, second_layer_dim, latent_dim):
    mae = Autoencoder(input_dim, first_layer_dim, second_layer_dim, latent_dim)
    mae_state = pickle.load(open(filepath, 'rb'))
    
    if isinstance(mae_state, list) and len(mae_state) > 0:
        mae_state = mae_state[0]  # Extract the first dictionary if it's a list of dictionaries
    
    # Convert numpy arrays to PyTorch tensors
    if isinstance(mae_state, dict):
        for key in mae_state:
            if isinstance(mae_state[key], np.ndarray):
                mae_state[key] = torch.tensor(mae_state[key])
        mae.load_state_dict(mae_state)
    else:
        print("Unexpected data structure:", type(mae_state), mae_state)
    
    return mae
def train_model(model, train_loader, val_loader, num_epoch, patience, learning_rate, split_num):
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
        val_loss = 0.0
        predictions = []
        targets_list = []
        with torch.no_grad():
            for batch in val_loader:
                inputs = [tensor.to(device) for tensor in batch[:-1]]
                targets = batch[-1].to(device)
                outputs = model(*inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                predictions.extend(outputs.cpu().numpy())
                targets_list.extend(targets.cpu().numpy())

        val_loss /= len(val_loader)
        print(f"Val Loss: {val_loss}")

        predictions = np.array(predictions).flatten()
        targets = np.array(targets_list).flatten()
        pearson_corr, _ = pearsonr(predictions, targets)
        print(f"Pearson Correlation: {pearson_corr}")

        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_pearson_correlation": pearson_corr,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "batch_size": train_loader.batch_size,
            "epoch": epoch + 1
        })

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            best_model_state_dict = model.state_dict()
            torch.save(best_model_state_dict, f'PytorchStaticSplits/OriginalCode/Results/Split{split_num}/PredictionNetworkModels/best_model_ae_split_{split_num}.pth')
            print("Model saved")

    return best_model_state_dict, training_predictions, training_targets_list

if __name__ == '__main__':

    split_num = 2

    ccl_size = "278"
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # with open('Data/ccl_complete_data_278CCL_1298DepOI_360844samples.pickle', 'rb') as f:
    #     data_mut, data_exp, data_cna, data_meth, data_dep, data_fprint = pickle.load(f)

    with open(f'Data/data_split_{split_num}.pickle', 'rb') as f:
        train_dataset, val_dataset, test_dataset = pickle.load(f)

    run = wandb.init(project="Self-Supervised-Machine-Learning-For-Predicting-Cancer-Dependencies-Splits", entity="kemal-bayik", name=f"Just_NN_{ccl_size}CCL_{current_time}_AE_Split_{split_num}")

    config = wandb.config
    config.learning_rate = 1e-4
    config.batch_size = 10000
    config.epochs = 100
    config.patience = 3

    latent_dim = 50

    # Define dimensions for the pretrained MAEs
    dims_mut = (train_dataset[:][0].shape[1], 1000, 100, 50)
    dims_exp = (train_dataset[:][1].shape[1], 500, 200, 50)
    dims_cna = (train_dataset[:][2].shape[1], 500, 200, 50)
    dims_meth = (train_dataset[:][3].shape[1], 500, 200, 50)
    dims_fprint = (train_dataset[:][4].shape[1], 1000, 100, 50)

    # Load pre-trained MAE models    
    premodel_mut = load_pretrained_ae(f'PytorchStaticSplits/OriginalCode/Results/Split{split_num}/USL_Pretrained/tcga_mut_ae_best_split_{split_num}.pickle', *dims_mut)
    premodel_exp = load_pretrained_ae(f'PytorchStaticSplits/OriginalCode/Results/Split{split_num}/USL_Pretrained/tcga_exp_ae_best_split_{split_num}.pickle', *dims_exp)
    premodel_cna = load_pretrained_ae(f'PytorchStaticSplits/OriginalCode/Results/Split{split_num}/USL_Pretrained/tcga_cna_ae_best_split_{split_num}.pickle', *dims_cna)
    premodel_meth = load_pretrained_ae(f'PytorchStaticSplits/OriginalCode/Results/Split{split_num}/USL_Pretrained/tcga_meth_ae_best_split_{split_num}.pickle', *dims_meth)
    premodel_fprint = Autoencoder(input_dim=train_dataset[:][4].shape[1], first_layer_dim=1000, second_layer_dim=100, latent_dim=50)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Create the DeepDEP model using the pretrained MAE models
    model = DeepDEP(premodel_mut, premodel_exp, premodel_cna, premodel_meth, premodel_fprint, latent_dim, 250)
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

    # # Save the best model
    # torch.save(best_model_state_dict, 'results/models/deepdep_mae_model_last.pth')
    
    # # Plot results
    y_true_train = np.array(training_targets_list).flatten()
    y_pred_train = np.array(training_predictions).flatten()
    y_true_test = np.array(targets_list).flatten()
    y_pred_test = np.array(predictions).flatten()

    np.savetxt(f'PytorchStaticSplits/OriginalCode/Results/Split{split_num}/PredictionNetworkModels/Predictions/y_true_train_CCL_AE_Split_{split_num}.txt', y_true_train, fmt='%.6f')
    np.savetxt(f'PytorchStaticSplits/OriginalCode/Results/Split{split_num}/PredictionNetworkModels/Predictions/y_pred_train_CCL_AE_Split_{split_num}.txt', y_pred_train, fmt='%.6f')
    np.savetxt(f'PytorchStaticSplits/OriginalCode/Results/Split{split_num}/PredictionNetworkModels/Predictions/y_true_test_CCL_AE_Split_{split_num}.txt', y_true_test, fmt='%.6f')
    np.savetxt(f'PytorchStaticSplits/OriginalCode/Results/Split{split_num}/PredictionNetworkModels/Predictions/y_pred_test_CCL_AE_Split_{split_num}.txt', y_pred_test, fmt='%.6f')

    print(f"Training: y_true_train size: {len(y_true_train)}, y_pred_train size: {len(y_pred_train)}")
    print(f"Testing: y_true_test size: {len(y_true_test)}, y_pred_test size: {len(y_pred_test)}")

    #plot_results(y_true_train, y_pred_train, y_true_test, y_pred_test, config.batch_size, config.learning_rate, config.epochs)
    #plot_density(y_true_train, y_pred_train, y_true_test, y_pred_test, config.batch_size, config.learning_rate, config.epochs)

    run.finish()
