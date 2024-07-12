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

        return reconstructed, mask, latent

class DeepDEP(nn.Module):
    def __init__(self, premodel_mut, premodel_exp, premodel_cna, premodel_meth, premodel_fprint, latent_dim, dense_layer_dim):
        super(DeepDEP, self).__init__()
        self.mae_mut = premodel_mut
        self.mae_exp = premodel_exp
        self.mae_cna = premodel_cna
        self.mae_meth = premodel_meth
        self.mae_fprint = premodel_fprint

        # Calculate the total dimension after concatenating latent dimensions
        latent_dim_total = latent_dim * 5  # 5 autoencoders with latent_dim each
        self.fc_merged1 = nn.Linear(latent_dim_total, dense_layer_dim)
        self.fc_merged2 = nn.Linear(dense_layer_dim, dense_layer_dim)
        self.fc_out = nn.Linear(dense_layer_dim, 1)

    def forward(self, mut, exp, cna, meth, fprint):
        recon_mut, mask_mut, latent_mut = self.mae_mut(mut)
        recon_exp, mask_exp, latent_exp = self.mae_exp(exp)
        recon_cna, mask_cna, latent_cna = self.mae_cna(cna)
        recon_meth, mask_meth, latent_meth = self.mae_meth(meth)
        recon_fprint, mask_fprint, latent_fprint = self.mae_fprint(fprint)
        
        merged = torch.cat([latent_mut, latent_exp, latent_cna, latent_meth, latent_fprint], dim=1)
        merged = torch.relu(self.fc_merged1(merged))
        merged = torch.relu(self.fc_merged2(merged))
        output = self.fc_out(merged)
        return output

def load_pretrained_mae(filepath, input_dim, first_layer_dim, second_layer_dim, latent_dim):
    mae = MaskedAutoencoder(input_dim, first_layer_dim, second_layer_dim, latent_dim)
    mae_state = pickle.load(open(filepath, 'rb'))
    
    # Convert numpy arrays to PyTorch tensors
    for key in mae_state:
        if isinstance(mae_state[key], np.ndarray):
            mae_state[key] = torch.tensor(mae_state[key])
    
    mae.load_state_dict(mae_state)
    return mae

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
    with open('Data/ccl_complete_data_278CCL_1298DepOI_360844samples.pickle', 'rb') as f:
        data_mut, data_exp, data_cna, data_meth, data_dep, data_fprint = pickle.load(f)

    wandb.init(project="Self-Supervised-Machine-Learning-For-Predicting-Cancer-Dependencies", entity="kemal-bayik", name=f"Just_NN_{ccl_size}CCL_{current_time}_MAE")

    config = wandb.config
    config.learning_rate = 1e-4
    config.batch_size = 10000
    config.epochs = 100
    config.patience = 3

    latent_dim = 50

    # Define dimensions for the pretrained MAEs
    dims_mut = (data_mut.shape[1], 1000, 100, 50)
    dims_exp = (data_exp.shape[1], 500, 200, 50)
    dims_cna = (data_cna.shape[1], 500, 200, 50)
    dims_meth = (data_meth.shape[1], 500, 200, 50)
    dims_fprint = (data_fprint.shape[1], 1000, 100, 50)

    # Load pre-trained MAE models    
    premodel_mut = load_pretrained_mae('results/masked_autoencoders/USL_pretrained/premodel_tcga_mut_mae_best.pickle', *dims_mut)
    premodel_exp = load_pretrained_mae('results/masked_autoencoders/USL_pretrained/premodel_tcga_exp_mae_best.pickle', *dims_exp)
    premodel_cna = load_pretrained_mae('results/masked_autoencoders/USL_pretrained/premodel_tcga_cna_mae_best.pickle', *dims_cna)
    premodel_meth = load_pretrained_mae('results/masked_autoencoders/USL_pretrained/premodel_tcga_meth_mae_best.pickle', *dims_meth)
    premodel_fprint = MaskedAutoencoder(input_dim=data_fprint.shape[1], first_layer_dim=1000, second_layer_dim=100, latent_dim=50)

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

    # Create the DeepDEP model using the pretrained MAE models
    model = DeepDEP(premodel_mut, premodel_exp, premodel_cna, premodel_meth, premodel_fprint, latent_dim, 250)
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
    torch.save(best_model_state_dict, 'results/models/deepdep_mae_model.pth')
    
    # Plot results
    y_true_train = np.array(training_targets_list).flatten()
    y_pred_train = np.array(training_predictions).flatten()
    y_true_test = np.array(targets_list).flatten()
    y_pred_test = np.array(predictions).flatten()

    np.savetxt(f'results/predictions/y_true_train_CCL_MAE.txt', y_true_train, fmt='%.6f')
    np.savetxt(f'results/predictions/y_pred_train_CCL_MAE.txt', y_pred_train, fmt='%.6f')
    np.savetxt(f'results/predictions/y_true_test_CCL_MAE.txt', y_true_test, fmt='%.6f')
    np.savetxt(f'results/predictions/y_pred_test_CCL_MAE.txt', y_pred_test, fmt='%.6f')

    print(f"Training: y_true_train size: {len(y_true_train)}, y_pred_train size: {len(y_pred_train)}")
    print(f"Testing: y_true_test size: {len(y_true_test)}, y_pred_test size: {len(y_pred_test)}")

    #plot_results(y_true_train, y_pred_train, y_true_test, y_pred_test, config.batch_size, config.learning_rate, config.epochs)
    #plot_density(y_true_train, y_pred_train, y_true_test, y_pred_test, config.batch_size, config.learning_rate, config.epochs)
