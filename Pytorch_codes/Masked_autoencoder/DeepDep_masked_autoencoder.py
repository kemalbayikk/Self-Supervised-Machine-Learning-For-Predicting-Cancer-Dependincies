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
# device = "cuda"
print(device)

class MaskedAutoencoder(nn.Module):
    def __init__(self, input_dim, first_layer_dim, second_layer_dim, latent_dim):
        super(MaskedAutoencoder, self).__init__()
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

    def forward(self, x, mask):
        masked_x = x * mask
        encoded = self.encode(masked_x)
        decoded = self.decode(encoded)
        return decoded

def masked_autoencoder_loss_function(recon_x, x, mask):
    masked_recon_x = recon_x * (1 - mask)  # Only consider masked parts
    masked_x = x * (1 - mask)  # Only consider masked parts
    recon_loss = nn.functional.mse_loss(masked_recon_x, masked_x, reduction='sum') / torch.sum(1 - mask)  # MSE loss over masked parts
    return recon_loss

class DeepDEP(nn.Module):
    def __init__(self, premodel_mut, premodel_exp, premodel_cna, premodel_meth, fprint_dim, dense_layer_dim):
        super(DeepDEP, self).__init__()
        self.mae_mut = premodel_mut
        self.mae_exp = premodel_exp
        self.mae_cna = premodel_cna
        self.mae_meth = premodel_meth

        self.mae_gene = MaskedAutoencoder(fprint_dim, 1000, 100, 50)

        # Update the input dimension of fc_merged1 to match the concatenated tensor's dimension
        input_dim = 50 * 5  # 50 is the latent dimension of each autoencoder
        self.fc_merged1 = nn.Linear(input_dim, dense_layer_dim)
        self.fc_merged2 = nn.Linear(dense_layer_dim, dense_layer_dim)
        self.fc_out = nn.Linear(dense_layer_dim, 1)

    def forward(self, mut, exp, cna, meth, fprint, mask_mut, mask_exp, mask_cna, mask_meth, mask_fprint):
        encoded_mut = self.mae_mut.encode(mut * mask_mut)
        encoded_exp = self.mae_exp.encode(exp * mask_exp)
        encoded_cna = self.mae_cna.encode(cna * mask_cna)
        encoded_meth = self.mae_meth.encode(meth * mask_meth)
        encoded_gene = self.mae_gene.encode(fprint * mask_fprint)

        merged = torch.cat([encoded_mut, encoded_exp, encoded_cna, encoded_meth, encoded_gene], dim=1)
        merged = torch.relu(self.fc_merged1(merged))
        merged = torch.relu(self.fc_merged2(merged))
        output = self.fc_out(merged)

        recon_mut = self.mae_mut.decode(encoded_mut)
        recon_exp = self.mae_exp.decode(encoded_exp)
        recon_cna = self.mae_cna.decode(encoded_cna)
        recon_meth = self.mae_meth.decode(encoded_meth)
        recon_gene = self.mae_gene.decode(encoded_gene)

        return output, recon_mut, mut, recon_exp, exp, recon_cna, cna, recon_meth, meth, recon_gene, fprint

def total_loss_function(outputs, targets, masks):
    output, recon_mut, mut, recon_exp, exp, recon_cna, cna, recon_meth, meth, recon_gene, fprint = outputs
    mask_mut, mask_exp, mask_cna, mask_meth, mask_fprint = masks

    recon_loss_mut = masked_autoencoder_loss_function(recon_mut, mut, mask_mut)
    recon_loss_exp = masked_autoencoder_loss_function(recon_exp, exp, mask_exp)
    recon_loss_cna = masked_autoencoder_loss_function(recon_cna, cna, mask_cna)
    recon_loss_meth = masked_autoencoder_loss_function(recon_meth, meth, mask_meth)
    recon_loss_gene = masked_autoencoder_loss_function(recon_gene, fprint, mask_fprint)

    return nn.functional.mse_loss(output, targets) + recon_loss_mut + recon_loss_exp + recon_loss_cna + recon_loss_meth + recon_loss_gene


def load_pretrained_mae(filepath, input_dim, first_layer_dim, second_layer_dim, latent_dim):
    mae = MaskedAutoencoder(input_dim, first_layer_dim, second_layer_dim, latent_dim)
    mae_state = pickle.load(open(filepath, 'rb'))
    
    # Convert numpy arrays to PyTorch tensors
    for key in mae_state:
        if isinstance(mae_state[key], np.ndarray):
            mae_state[key] = torch.tensor(mae_state[key])
    
    mae.load_state_dict(mae_state)
    return mae

def generate_mask(shape, mask_ratio=0.15):
    mask = torch.ones(shape)
    num_masked = int(mask_ratio * shape[1])
    for i in range(shape[0]):
        indices = np.random.choice(shape[1], num_masked, replace=False)
        mask[i, indices] = 0
    return mask.to(device)

def train_model(model, train_loader, test_loader, num_epoch, patience, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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

            # Generate masks for each input type
            mask_mut = generate_mask(inputs[0].shape)
            mask_exp = generate_mask(inputs[1].shape)
            mask_cna = generate_mask(inputs[2].shape)
            mask_meth = generate_mask(inputs[3].shape)
            mask_fprint = generate_mask(inputs[4].shape)

            optimizer.zero_grad()
            outputs = model(*inputs, mask_mut, mask_exp, mask_cna, mask_meth, mask_fprint)
            masks = (mask_mut, mask_exp, mask_cna, mask_meth, mask_fprint)
            loss = total_loss_function(outputs, targets, masks)
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

                # Generate masks for each input type
                mask_mut = generate_mask(inputs[0].shape)
                mask_exp = generate_mask(inputs[1].shape)
                mask_cna = generate_mask(inputs[2].shape)
                mask_meth = generate_mask(inputs[3].shape)
                mask_fprint = generate_mask(inputs[4].shape)
                
                outputs = model(*inputs, mask_mut, mask_exp, mask_cna, mask_meth, mask_fprint)
                masks = (mask_mut, mask_exp, mask_cna, mask_meth, mask_fprint)
                loss = total_loss_function(outputs, targets, masks)
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

        # Save the best model
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

    return best_model_state_dict  # Return the best model's state_dict


if __name__ == '__main__':
    ccl_size = "28"
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    with open('Data/ccl_complete_data_28CCL_1298DepOI_36344samples_demo.pickle', 'rb') as f:
        data_mut, data_exp, data_cna, data_meth, data_dep, data_fprint = pickle.load(f)

    wandb.init(project="Self-Supervised-Machine-Learning-For-Predicting-Cancer-Dependencies", entity="kemal-bayik", name=f"MaskedAE_DeepDEP_{ccl_size}CCL_{current_time}")

    config = wandb.config
    config.learning_rate = 1e-4
    config.batch_size = 256
    config.epochs = 100
    config.patience = 5

    # Define dimensions for the pretrained MAEs
    dims_mut = (data_mut.shape[1], 1000, 100, 50)
    dims_exp = (data_exp.shape[1], 500, 200, 50)
    dims_cna = (data_cna.shape[1], 500, 200, 50)
    dims_meth = (data_meth.shape[1], 500, 200, 50)

    # Load pre-trained MAE models
    premodel_mut = load_pretrained_mae('results/masked_autoencoders/premodel_tcga_mut_masked_autoencoder.pickle', *dims_mut)
    premodel_exp = load_pretrained_mae('results/masked_autoencoders/premodel_tcga_exp_masked_autoencoder.pickle', *dims_exp)
    premodel_cna = load_pretrained_mae('results/masked_autoencoders/premodel_tcga_cna_masked_autoencoder.pickle', *dims_cna)
    premodel_meth = load_pretrained_mae('results/masked_autoencoders/premodel_tcga_meth_masked_autoencoder.pickle', *dims_meth)

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

    # Create the DeepDEP model using the pretrained MAE models
    model = DeepDEP(premodel_mut, premodel_exp, premodel_cna, premodel_meth, data_fprint.shape[1], 250)
    best_model_state_dict = train_model(model, train_loader, test_loader, config.epochs, config.patience, config.learning_rate)

    # Load the best model and calculate Pearson Correlation
    model.load_state_dict(best_model_state_dict)
    model.eval()
    predictions = []
    targets_list = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = [tensor.to(device) for tensor in batch[:-1]]
            targets = batch[-1].to(device)

            # Generate masks for each input type
            mask_mut = generate_mask(inputs[0].shape)
            mask_exp = generate_mask(inputs[1].shape)
            mask_cna = generate_mask(inputs[2].shape)
            mask_meth = generate_mask(inputs[3].shape)
            mask_fprint = generate_mask(inputs[4].shape)
            
            outputs = model(*inputs, mask_mut, mask_exp, mask_cna, mask_meth, mask_fprint)
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
    torch.save(best_model_state_dict, 'model_demo_masked_autoencoder.pth')
