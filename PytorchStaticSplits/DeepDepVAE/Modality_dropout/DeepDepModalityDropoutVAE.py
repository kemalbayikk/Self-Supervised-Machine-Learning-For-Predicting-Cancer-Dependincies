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

    def forward(self, mut, exp, cna, meth, fprint, p_drop=0.0, drop_mask=None):
        recon_mut, mu_mut, logvar_mut = self.vae_mut(mut)
        recon_exp, mu_exp, logvar_exp = self.vae_exp(exp)
        recon_cna, mu_cna, logvar_cna = self.vae_cna(cna)
        recon_meth, mu_meth, logvar_meth = self.vae_meth(meth)
        recon_gene, mu_fprint, logvar_gene = self.vae_fprint(fprint)
        
        # Apply input dropout using the drop_mask
        if drop_mask is not None:
            mu_mut = self.apply_dropout(mu_mut, p_drop, drop_mask[0])
            mu_exp = self.apply_dropout(mu_exp, p_drop, drop_mask[1])
            mu_cna = self.apply_dropout(mu_cna, p_drop, drop_mask[2])
            mu_meth = self.apply_dropout(mu_meth, p_drop, drop_mask[3])
            mu_fprint = self.apply_dropout(mu_fprint, p_drop, drop_mask[4])
        
        merged = torch.cat([mu_mut, mu_exp, mu_cna, mu_meth, mu_fprint], dim=1)
        merged = torch.relu(self.fc_merged1(merged))
        merged = torch.relu(self.fc_merged2(merged))
        output = self.fc_out(merged)
        return output
    
    def apply_dropout(self, x, p_drop, drop):
        if drop:
            return torch.zeros_like(x)
        return x


def load_pretrained_vae(filepath, input_dim, first_layer_dim, second_layer_dim, latent_dim):
    vae = VariationalAutoencoder(input_dim, first_layer_dim, second_layer_dim, latent_dim)
    vae_state = pickle.load(open(filepath, 'rb'))
    
    # Convert numpy arrays to PyTorch tensors
    for key in vae_state:
        if isinstance(vae_state[key], np.ndarray):
            vae_state[key] = torch.tensor(vae_state[key])
    
    vae.load_state_dict(vae_state)
    return vae

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
            # drop_mask = [torch.rand(1).item() < p_drop for _ in range(4)] + [False]
            drop_mask = [torch.rand(1).item() < p_drop for _ in range(5)]
            # drop_mask = [0,0,0,0,1]

            # Drop maskesi bilgisini yazdır
            # masked_vaes = ['mut', 'exp', 'cna', 'meth', 'fprint']
            # masked_vaes = [vae for vae, mask in zip(masked_vaes, drop_mask) if mask]
            # print(f"Epoch {epoch+1}, Masked VAEs: {masked_vaes}")
            
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
            "val_pearson_correlation": pearson_corr,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "batch_size": train_loader.batch_size,
            "epoch": epoch + 1
        })

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            best_model_state_dict = model.state_dict()
            torch.save(best_model_state_dict, 'PytorchStaticSplits/DeepDepVAE/Results/Split2/ModalityDropout/best_model_vae_modality_dropout_newVAE_withsplit2_Best_Model.pth')
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
        [0, 0, 0, 0, 0],  # Hiçbiri kapalı değil (referans)
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 1, 1],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 1],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 1, 0],
        [1, 0, 0, 1, 1],
        [1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1],
        [1, 0, 1, 1, 0],
        [1, 0, 1, 1, 1],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 1],
        [1, 1, 0, 1, 0],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1]
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

    split_num = 2

    with open(f'Data/data_split_{split_num}.pickle', 'rb') as f:
            train_dataset, val_dataset, test_dataset = pickle.load(f)

    wandb.init(project="VAE-PredictionNetwork-BestModel-ModalityDropout", entity="kemal-bayik", name=f"Prediction_Network_Modality_Dropout_VAE_withsplit{split_num}_All_Modalities")

    config = wandb.config
    config.learning_rate = 1e-3
    config.batch_size = 10000
    config.epochs = 100
    config.patience = 3
    config.p_drop = 0.5  # Dropout olasılığı

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
    model = DeepDEP(premodel_mut, premodel_exp, premodel_cna, premodel_meth, premodel_fprint, 250)
    best_model_state_dict, training_predictions, training_targets_list = train_model(model, train_loader, val_loader, config.epochs, config.patience, config.learning_rate, config.p_drop )
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
        np.savetxt(f'PytorchStaticSplits/DeepDepVAE/Results/Split2/ModalityDropout/Predictions/y_true_test_mask_{mask}_VAE_Modality_Dropout_NewVAE_withsplit2_Best_Model.txt', res['targets'], fmt='%.6f')
        np.savetxt(f'PytorchStaticSplits/DeepDepVAE/Results/Split2/ModalityDropout/Predictions/y_pred_test_mask_{mask}_VAE_Modality_Dropout_NewVAE_withsplit2_Best_Model.txt', res['predictions'], fmt='%.6f')

    # Save the best model
    torch.save(best_model_state_dict, 'PytorchStaticSplits/DeepDepVAE/Results/Split2/ModalityDropout/deepdep_vae_model_modality_dropout_NewVAE_withsplit2_Best_Model.pth')

    # Plot results
    y_true_train = np.array(training_targets_list).flatten()
    y_pred_train = np.array(training_predictions).flatten()
    y_true_test = results["0_0_0_0_0"]["targets"].flatten()  # Hiçbir modalite kapalı değilken
    y_pred_test = results["0_0_0_0_0"]["predictions"].flatten()  # Hiçbir modalite kapalı değilken

    np.savetxt(f'PytorchStaticSplits/DeepDepVAE/Results/Split2/ModalityDropout/Predictions/y_true_train_VAE_Modality_Dropout_NewVAE_withsplit2_Best_Model.txt', y_true_train, fmt='%.6f')
    np.savetxt(f'PytorchStaticSplits/DeepDepVAE/Results/Split2/ModalityDropout/Predictions/y_pred_train_VAE_Modality_Dropout_NewVAE_withsplit2_Best_Model.txt', y_pred_train, fmt='%.6f')
    np.savetxt(f'PytorchStaticSplits/DeepDepVAE/Results/Split2/ModalityDropout/Predictions/y_true_test_VAE_Modality_Dropout_NewVAE_withsplit2_Best_Model.txt', y_true_test, fmt='%.6f')
    np.savetxt(f'PytorchStaticSplits/DeepDepVAE/Results/Split2/ModalityDropout/Predictions/y_pred_test_VAE_Modality_Dropout_NewVAE_withsplit2_Best_Model.txt', y_pred_test, fmt='%.6f')

    print(f"Training: y_true_train size: {len(y_true_train)}, y_pred_train size: {len(y_pred_train)}")
    print(f"Testing: y_true_test size: {len(y_true_test)}, y_pred_test size: {len(y_pred_test)}")
