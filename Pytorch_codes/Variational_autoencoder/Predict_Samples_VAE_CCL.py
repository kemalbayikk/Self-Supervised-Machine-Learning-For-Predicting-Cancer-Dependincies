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
    def __init__(self, dims_mut, dims_exp, dims_cna, dims_meth, fprint_dim, dense_layer_dim):
        super(DeepDEP, self).__init__()
        self.vae_mut = VariationalAutoencoder(dims_mut, 1000, 100, 50)
        self.vae_exp = VariationalAutoencoder(dims_exp, 500, 200, 50)
        self.vae_cna = VariationalAutoencoder(dims_cna, 500, 200, 50)
        self.vae_meth = VariationalAutoencoder(dims_meth, 500, 200, 50)

        self.vae_fprint = VariationalAutoencoder(fprint_dim, 1000, 100, 50)

        self.fc_merged1 = nn.Linear(250, dense_layer_dim)
        self.fc_merged2 = nn.Linear(dense_layer_dim, dense_layer_dim)
        self.fc_out = nn.Linear(dense_layer_dim, 1)

    def forward(self, mut, exp, cna, meth, fprint):
        recon_mut, mu_mut, logvar_mut = self.vae_mut(mut)
        recon_exp, mu_exp, logvar_exp = self.vae_exp(exp)
        recon_cna, mu_cna, logvar_cna = self.vae_cna(cna)
        recon_meth, mu_meth, logvar_meth = self.vae_meth(meth)
        recon_fprint, mu_fprint, logvar_fprint = self.vae_fprint(fprint)
        
        merged = torch.cat([mu_mut, mu_exp, mu_cna, mu_meth, mu_fprint], dim=1)
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
    plt.title(f'Density plot of Dependency Scores\nBatch Size: {batch_size}, Learning Rate: {learning_rate}, Epochs: {epochs} VAE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'PytorchStaticSplits/DeepDepVAE/Analysis/dependency_score_density_plot_{batch_size}_{learning_rate}_{epochs}_VAE_last.png')
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
    plt.title(f'Training/validation\nBatch Size: {batch_size}, Learning Rate: {learning_rate}, Epochs: {epochs}, VAE')
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
    plt.title(f'Testing\nBatch Size: {batch_size}, Learning Rate: {learning_rate}, Epochs: {epochs}, VAE')
    plt.xlim(-4, 5)
    plt.ylim(-4, 5)
    pearson_corr_test, _ = pearsonr(y_pred_test, y_true_test)
    mse_test = mean_squared_error(y_true_test, y_pred_test)
    plt.text(0.1, 0.9, f'$\\rho$ = {pearson_corr_test:.2f}\nMSE = {mse_test:.3f}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.8, f'y = {coef_test[0]:.2f}x + {coef_test[1]:.2f}', color='red', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.savefig(f'PytorchStaticSplits/DeepDepVAE/Analysis/prediction_scatter_plots_{batch_size}_{learning_rate}_{epochs}_VAE.png')
    plt.show()

if __name__ == '__main__':
    device = "mps"
    
    # Define the model architecture with correct dimensions
    dims_mut = 4539  # Correct dimension based on the error message
    dims_exp = 6016
    dims_cna = 7460
    dims_meth = 6617
    fprint_dim = 3115  # Correct dimension based on the error message
    dense_layer_dim = 250

    model = DeepDEP(dims_mut, dims_exp, dims_cna, dims_meth, fprint_dim, dense_layer_dim).to(device)

    # Load the PyTorch model state dictionary
    model.load_state_dict(torch.load(f"PytorchStaticSplits/DeepDepVAE/Results/Split2/PredictionNetworkModels/VAE_Prediction_Network_Split_2_LR_0.001.pth", map_location=device))
    model.eval()

    # Load TCGA genomics data and gene fingerprints
    data_mut_tcga, data_labels_mut_tcga, sample_names_mut_tcga, gene_names_mut_tcga = load_data("Data/CCL/ccl_mut_data_paired_with_tcga.txt")
    data_exp_tcga, data_labels_exp_tcga, sample_names_exp_tcga, gene_names_exp_tcga = load_data("Data/CCL/ccl_exp_data_paired_with_tcga.txt")
    data_cna_tcga, data_labels_cna_tcga, sample_names_cna_tcga, gene_names_cna_tcga = load_data("Data/CCL/ccl_cna_data_paired_with_tcga.txt")
    data_meth_tcga, data_labels_meth_tcga, sample_names_meth_tcga, gene_names_meth_tcga = load_data("Data/CCL/ccl_meth_data_paired_with_tcga.txt")
    data_fprint_1298DepOIs, data_labels_fprint, gene_names_fprint, function_names_fprint = load_data("Data/crispr_gene_fingerprint_cgp.txt")
    print("\n\nDatasets successfully loaded.\n\n")

    batch_size = 10000
    first_to_predict = 278
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
        print("CCL sample %d predicted..." % z)

    # y_true_train = np.loadtxt('Pytorch_codes/Variational_autoencoder/Models To Analyze/Split 2 VAE/y_true_train_CCL_VAE_Split_2.txt', dtype=float)
    # y_pred_train = np.loadtxt('Pytorch_codes/Variational_autoencoder/Models To Analyze/Split 2 VAE/y_pred_train_CCL_VAE_Split_2.txt', dtype=float)
    # y_true_test = np.loadtxt('Pytorch_codes/Variational_autoencoder/Models To Analyze/Split 2 VAE/y_true_test_CCL_VAE_Split_2.txt', dtype=float)
    # y_pred_test = np.loadtxt('Pytorch_codes/Variational_autoencoder/Models To Analyze/Split 2 VAE/y_pred_test_CCL_VAE_Split_2.txt', dtype=float)

    # Write prediction results to txt
    data_pred_df = pd.DataFrame(data=np.transpose(data_pred), index=gene_names_fprint, columns=sample_names_mut_tcga[0:first_to_predict])
    data_pred_df.to_csv(f"PytorchStaticSplits/DeepDepVAE/Analysis/ccl_predicted_data.txt", sep='\t', index_label='CRISPR_GENE', float_format='%.4f')
    # print("\n\nPrediction completed in %.1f mins.\nResults saved in /results/predictions/tcga_predicted_data_%s.txt\n\n" % ((time.time()-t)/60, model_name))

    # plot_density(y_true_train[0:len(y_true_train) - 1].flatten(),y_pred_train[0:len(y_pred_train) - 1].flatten(),data_pred.flatten(),batch_size,1e-4,100)
    # plot_results(y_true_train, y_pred_train, y_true_test, y_pred_test, batch_size, 1e-4, 100)
