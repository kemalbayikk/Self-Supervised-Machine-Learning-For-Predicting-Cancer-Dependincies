import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import seaborn as sns
import pickle

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

def load_data(filename):
    data = []
    gene_names = []
    data_labels = []
    lines = open(filename).readlines()
    sample_names = lines[0].replace('\n', '').split('\t')[1:]
    dx = 1

    for line in lines[dx:]:
        values = line.replace('\n', '').split('\t')
        gene = str.upper(values[0])
        gene_names.append(gene)
        data.append(values[1:])
    data = np.array(data, dtype='float32')
    data = np.transpose(data)

    return data, data_labels, sample_names, gene_names

def plot_density(y_true_train, y_pred_train, y_pred_test, batch_size, learning_rate, epochs):
    plt.figure(figsize=(8, 6))
    sns.kdeplot(y_true_train, label='CCL original', color='blue')
    sns.kdeplot(y_pred_train, label='CCL predicted', color='orange')
    sns.kdeplot(y_pred_test, label='Tumor predicted', color='brown')
    plt.xlabel('Dependency score')
    plt.ylabel('Density (x0.01)')
    plt.title(f'Density plot of Dependency Scores\nBatch Size: {batch_size}, Learning Rate: {learning_rate}, Epochs: {epochs} MAE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/predictions/dependency_score_density_plot_{batch_size}_{learning_rate}_{epochs}_MAE.png')
    plt.show()

def plot_results(y_true_train, y_pred_train, y_true_test, y_pred_test, batch_size, learning_rate, epochs):
    plt.figure(figsize=(14, 6))

    # Training/validation plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred_train, y_true_train, alpha=0.5)
    coef_train = np.polyfit(y_pred_train, y_true_train, 1)
    poly1d_fn_train = np.poly1d(coef_train)
    plt.plot(np.unique(y_pred_train), poly1d_fn_train(np.unique(y_pred_train)), color='red')
    plt.xlabel('DeepDEP-predicted score')
    plt.ylabel('Original dependency score')
    plt.title(f'Training/validation\nBatch Size: {batch_size}, Learning Rate: {learning_rate}, Epochs: {epochs}, MAE')
    plt.xlim(-4, 5)
    plt.ylim(-4, 5)
    pearson_corr_train, _ = pearsonr(y_pred_train, y_true_train)
    mse_train = mean_squared_error(y_true_train, y_pred_train)
    plt.text(0.1, 0.9, f'$\\rho$ = {pearson_corr_train:.2f}\nMSE = {mse_train:.3f}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.8, f'y = {coef_train[0]:.2f}x + {coef_train[1]:.2f}', color='red', transform=plt.gca().transAxes)

    # Testing plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred_test, y_true_test, alpha=0.5)
    coef_test = np.polyfit(y_pred_test, y_true_test, 1)
    poly1d_fn_test = np.poly1d(coef_test)
    plt.plot(np.unique(y_pred_test), poly1d_fn_test(np.unique(y_pred_test)), color='red')
    plt.xlabel('DeepDEP-predicted score')
    plt.ylabel('Original dependency score')
    plt.title(f'Testing\nBatch Size: {batch_size}, Learning Rate: {learning_rate}, Epochs: {epochs}, MAE')
    plt.xlim(-4, 5)
    plt.ylim(-4, 5)
    pearson_corr_test, _ = pearsonr(y_pred_test, y_true_test)
    mse_test = mean_squared_error(y_true_test, y_pred_test)
    plt.text(0.1, 0.9, f'$\\rho$ = {pearson_corr_test:.2f}\nMSE = {mse_test:.3f}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.8, f'y = {coef_test[0]:.2f}x + {coef_test[1]:.2f}', color='red', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.savefig(f'results/predictions/prediction_scatter_plots_{batch_size}_{learning_rate}_{epochs}_MAE.png')
    plt.show()

def load_pretrained_mae(filepath, input_dim, first_layer_dim, second_layer_dim, latent_dim):
    mae = MaskedAutoencoder(input_dim, first_layer_dim, second_layer_dim, latent_dim)
    mae_state = pickle.load(open(filepath, 'rb'))
    
    # Convert numpy arrays to PyTorch tensors
    for key in mae_state:
        if isinstance(mae_state[key], np.ndarray):
            mae_state[key] = torch.tensor(mae_state[key])
    
    mae.load_state_dict(mae_state)
    return mae

if __name__ == '__main__':
    model_name = "deepdep_mae_model"  # "model_paper"
    device = "mps"
    
    # Define the model architecture with correct dimensions
    # dims_mut = 4539  # Correct dimension based on the error message
    # dims_exp = 6016
    # dims_cna = 7460
    # dims_meth = 6617
    # fprint_dim = 3115  # Correct dimension based on the error message

        # Define dimensions for the pretrained MAEs
    dims_mut = (4539, 1000, 100, 50)
    dims_exp = (6016, 500, 200, 50)
    dims_cna = (7460, 500, 200, 50)
    dims_meth = (6617, 500, 200, 50)
    dims_fprint = (3115, 1000, 100, 50)
    dense_layer_dim = 250

    # Load pre-trained MAE models    
    premodel_mut = load_pretrained_mae('results/masked_autoencoders/USL_pretrained/premodel_tcga_mut_mae_best.pickle', *dims_mut)
    premodel_exp = load_pretrained_mae('results/masked_autoencoders/USL_pretrained/premodel_tcga_exp_mae_best.pickle', *dims_exp)
    premodel_cna = load_pretrained_mae('results/masked_autoencoders/USL_pretrained/premodel_tcga_cna_mae_best.pickle', *dims_cna)
    premodel_meth = load_pretrained_mae('results/masked_autoencoders/USL_pretrained/premodel_tcga_meth_mae_best.pickle', *dims_meth)
    premodel_fprint = MaskedAutoencoder(input_dim=3115, first_layer_dim=1000, second_layer_dim=100, latent_dim=50)
    
    model = DeepDEP(premodel_mut, premodel_exp, premodel_cna, premodel_meth, premodel_fprint, 50, 250).to(device)

    # Load the PyTorch model state dictionary
    model.load_state_dict(torch.load(f"results/models/{model_name}.pth", map_location=device))
    model.eval()

    # Load TCGA genomics data and gene fingerprints
    data_mut_tcga, data_labels_mut_tcga, sample_names_mut_tcga, gene_names_mut_tcga = load_data("Data/TCGA/tcga_mut_data_paired_with_ccl.txt")
    data_exp_tcga, data_labels_exp_tcga, sample_names_exp_tcga, gene_names_exp_tcga = load_data("Data/TCGA/tcga_exp_data_paired_with_ccl.txt")
    data_cna_tcga, data_labels_cna_tcga, sample_names_cna_tcga, gene_names_cna_tcga = load_data("Data/TCGA/tcga_cna_data_paired_with_ccl.txt")
    data_meth_tcga, data_labels_meth_tcga, sample_names_meth_tcga, gene_names_meth_tcga = load_data("Data/TCGA/tcga_meth_data_paired_with_ccl.txt")
    data_fprint_1298DepOIs, data_labels_fprint, gene_names_fprint, function_names_fprint = load_data("Data/crispr_gene_fingerprint_cgp.txt")
    print("\n\nDatasets successfully loaded.\n\n")

    batch_size = 10000
    first_to_predict = 8238
    data_pred = np.zeros((first_to_predict, data_fprint_1298DepOIs.shape[0]))
    
    t = time.time()
    for z in np.arange(0, first_to_predict):
        data_mut_batch = torch.tensor(data_mut_tcga[np.repeat(z, data_fprint_1298DepOIs.shape[0])], dtype=torch.float32).to(device)
        data_exp_batch = torch.tensor(data_exp_tcga[np.repeat(z, data_fprint_1298DepOIs.shape[0])], dtype=torch.float32).to(device)
        data_cna_batch = torch.tensor(data_cna_tcga[np.repeat(z, data_fprint_1298DepOIs.shape[0])], dtype=torch.float32).to(device)
        data_meth_batch = torch.tensor(data_meth_tcga[np.repeat(z, data_fprint_1298DepOIs.shape[0])], dtype=torch.float32).to(device)
        data_fprint_batch = torch.tensor(data_fprint_1298DepOIs, dtype=torch.float32).to(device)

        with torch.no_grad():
            data_pred_tmp = model(data_mut_batch, data_exp_batch, data_cna_batch, data_meth_batch, data_fprint_batch).cpu().numpy()
        
        data_pred[z] = np.transpose(data_pred_tmp)
        print("TCGA sample %d predicted..." % z)

    y_true_train = np.loadtxt('results/predictions/Masked Autoencoder/y_true_train_CCL_MAE.txt', dtype=float)
    y_pred_train = np.loadtxt('results/predictions/Masked Autoencoder/y_pred_train_CCL_MAE.txt', dtype=float)
    y_true_test = np.loadtxt('results/predictions/Masked Autoencoder/y_true_test_CCL_MAE.txt', dtype=float)
    y_pred_test = np.loadtxt('results/predictions/Masked Autoencoder/y_pred_test_CCL_MAE.txt', dtype=float)

    # Write prediction results to txt
    data_pred_df = pd.DataFrame(data=np.transpose(data_pred), index=gene_names_fprint, columns=sample_names_mut_tcga[0:first_to_predict])
    data_pred_df.to_csv(f"results/predictions/tcga_predicted_data_{model_name}.txt", sep='\t', index_label='CRISPR_GENE', float_format='%.4f')
    print("\n\nPrediction completed in %.1f mins.\nResults saved in /results/predictions/tcga_predicted_data_%s_mae.txt\n\n" % ((time.time()-t)/60, model_name))

    plot_density(y_true_train[0:len(y_true_train) - 1].flatten(),y_pred_train[0:len(y_pred_train) - 1].flatten(),data_pred.flatten(),10000,1e-4,100)
    plot_results(y_true_train, y_pred_train, y_true_test, y_pred_test, 10000, 1e-4, 100)
