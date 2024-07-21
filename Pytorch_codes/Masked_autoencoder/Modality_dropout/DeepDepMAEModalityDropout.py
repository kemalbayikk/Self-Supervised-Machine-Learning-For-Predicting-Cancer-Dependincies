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

    def forward(self, x):

        encoded = torch.relu(self.encoder_fc1(x))
        encoded = torch.relu(self.encoder_fc2(encoded))
        latent = self.encoder_fc3(encoded)

        return latent

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

    def forward(self, mut, exp, cna, meth, fprint, p_drop=0.0, drop_mask=None):
        latent_mut = self.mae_mut(mut)
        latent_exp = self.mae_exp(exp)
        latent_cna = self.mae_cna(cna)
        latent_meth = self.mae_meth(meth)
        latent_fprint = self.mae_fprint(fprint)

        # Apply input dropout using the drop_mask
        if drop_mask is not None:
            latent_mut = self.apply_dropout(latent_mut, drop_mask[0])
            latent_exp = self.apply_dropout(latent_exp, drop_mask[1])
            latent_cna = self.apply_dropout(latent_cna, drop_mask[2])
            latent_meth = self.apply_dropout(latent_meth, drop_mask[3])
            latent_fprint = self.apply_dropout(latent_fprint, drop_mask[4])
        
        merged = torch.cat([latent_mut, latent_exp, latent_cna, latent_meth, latent_fprint], dim=1)
        merged = torch.relu(self.fc_merged1(merged))
        merged = torch.relu(self.fc_merged2(merged))
        output = self.fc_out(merged)
        return output

    def apply_dropout(self, x, drop):
        if drop:
            return torch.zeros_like(x)
        return x

def load_pretrained_mae(filepath, input_dim, first_layer_dim, second_layer_dim, latent_dim):
    mae = MaskedAutoencoder(input_dim, first_layer_dim, second_layer_dim, latent_dim)
    mae_state = pickle.load(open(filepath, 'rb'))
    
    # Convert numpy arrays to PyTorch tensors
    for key in mae_state:
        if isinstance(mae_state[key], np.ndarray):
            mae_state[key] = torch.tensor(mae_state[key])
    
    mae.load_state_dict(mae_state)
    return mae

def train_model(model, train_loader, val_loader, num_epoch, patience, learning_rate, p_drop):
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
            
            # Dropout maskesi oluştur
            drop_mask = [torch.rand(1).item() < p_drop for _ in range(4)] + [False]
            
            # Drop maskesi bilgisini yazdır
            masked_vaes = ['mut', 'exp', 'cna', 'meth', 'fprint']
            masked_vaes = [vae for vae, mask in zip(masked_vaes, drop_mask) if mask]
            print(f"Epoch {epoch+1}, Masked VAEs: {masked_vaes}")

            outputs = model(*inputs, p_drop=p_drop, drop_mask=drop_mask)
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
                outputs = model(*inputs, p_drop=0.0, drop_mask=[0, 0, 0, 0, 0])  # Validation sırasında dropout yok
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                predictions.extend(outputs.cpu().numpy())
                targets_list.extend(targets.cpu().numpy())

        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss}")

        predictions = np.array(predictions).flatten()
        targets = np.array(targets_list).flatten()
        pearson_corr, _ = pearsonr(predictions, targets)
        print(f"Pearson Correlation: {pearson_corr}")

        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "pearson_correlation": pearson_corr,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "batch_size": train_loader.batch_size,
            "epoch": epoch + 1
        })

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            best_model_state_dict = model.state_dict()
            torch.save(best_model_state_dict, 'best_model_mae_input_dropout.pth')
            print("Model saved")

    return best_model_state_dict, training_predictions, training_targets_list


def test_model(model, test_loader, device):
    model.eval()
    drop_masks = [
        [1, 0, 0, 0, 0],  # mut kapalı
        [0, 1, 0, 0, 0],  # exp kapalı
        [0, 0, 1, 0, 0],  # cna kapalı
        [0, 0, 0, 1, 0],  # meth kapalı
        [0, 0, 0, 0, 1],  # fprint kapalı
        [0, 0, 0, 0, 0]   # Hiçbiri kapalı değil (referans)
    ]

    results = {}
    with torch.no_grad():
        for mask in drop_masks:
            predictions = []
            targets_list = []
            for batch in test_loader:
                inputs = [tensor.to(device) for tensor in batch[:-1]]
                targets = batch[-1].to(device)
                outputs = model(*inputs, p_drop=0.0, drop_mask=mask)
                predictions.extend(outputs.cpu().numpy())
                targets_list.extend(targets.cpu().numpy())

            predictions = np.array(predictions).flatten()
            targets = np.array(targets_list).flatten()
            pearson_corr, _ = pearsonr(predictions, targets)
            mask_str = "_".join(map(str, mask))
            results[mask_str] = {
                "predictions": predictions,
                "targets": targets,
                "pearson_corr": pearson_corr
            }
            print(f"Mask: {mask_str}, Pearson Correlation: {pearson_corr}")
    
    return results



if __name__ == '__main__':
    ccl_size = "278"
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open('Data/ccl_complete_data_278CCL_1298DepOI_360844samples.pickle', 'rb') as f:
        data_mut, data_exp, data_cna, data_meth, data_dep, data_fprint = pickle.load(f)

    wandb.init(project="Self-Supervised-Machine-Learning-For-Predicting-Cancer-Dependencies", entity="kemal-bayik", name=f"Just_NN_{ccl_size}CCL_{current_time}_MAE_input_dropout")

    config = wandb.config
    config.learning_rate = 1e-4
    config.batch_size = 10000
    config.epochs = 100
    config.patience = 3
    config.p_drop = 0.25  # Dropout olasılığı

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

    # Train/val/test split
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    print("Train size : ", train_size)
    print("Validation size : ", val_size)
    print("Test size : ", test_size)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Create the DeepDEP model using the pretrained MAE models
    model = DeepDEP(premodel_mut, premodel_exp, premodel_cna, premodel_meth, premodel_fprint, latent_dim, 250)
    best_model_state_dict, training_predictions, training_targets_list = train_model(model, train_loader, val_loader, config.epochs, config.patience, config.learning_rate, config.p_drop)

    # En iyi modeli yükleyip Pearson Korelasyonunu hesaplama
    model.load_state_dict(best_model_state_dict)
    
    # Test sırasında her bir modaliteyi kapatarak test etme
    results = test_model(model, test_loader, device)
    for mask, res in results.items():
        print(f"Mask: {mask}, Pearson Correlation: {res['pearson_corr']}")
        wandb.log({
            f"test_pearson_correlation_{mask}": res['pearson_corr'],
        })
        
        # Tahmin ve gerçek değerleri kaydetme
        np.savetxt(f'results/predictions/y_true_test_mask_{mask}.txt', res['targets'], fmt='%.6f')
        np.savetxt(f'results/predictions/y_pred_test_mask_{mask}.txt', res['predictions'], fmt='%.6f')

    # Save the best model
    torch.save(best_model_state_dict, 'results/models/deepdep_mae_model_input_dropout.pth')

    # Plot results
    y_true_train = np.array(training_targets_list).flatten()
    y_pred_train = np.array(training_predictions).flatten()
    y_true_test = results["0_0_0_0_0"]["targets"].flatten()  # Hiçbir modalite kapalı değilken
    y_pred_test = results["0_0_0_0_0"]["predictions"].flatten()  # Hiçbir modalite kapalı değilken

    np.savetxt(f'results/predictions/y_true_train_CCL_MAE_input_dropout.txt', y_true_train, fmt='%.6f')
    np.savetxt(f'results/predictions/y_pred_train_CCL_MAE_input_dropout.txt', y_pred_train, fmt='%.6f')
    np.savetxt(f'results/predictions/y_true_test_CCL_MAE_input_dropout.txt', y_true_test, fmt='%.6f')
    np.savetxt(f'results/predictions/y_pred_test_CCL_MAE_input_dropout.txt', y_pred_test, fmt='%.6f')

    print(f"Training: y_true_train size: {len(y_true_train)}, y_pred_train size: {len(y_pred_train)}")
    print(f"Testing: y_true_test size: {len(y_true_test)}, y_pred_test size: {len(y_pred_test)}")
