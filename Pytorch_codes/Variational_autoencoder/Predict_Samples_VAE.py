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

# class VAE(nn.Module):
#     def __init__(self, input_dim, hidden_dims, latent_dim):
#         super(VAE, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dims[0])
#         self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
#         self.fc3_mean = nn.Linear(hidden_dims[1], latent_dim)
#         self.fc3_log_var = nn.Linear(hidden_dims[1], latent_dim)

#     def encode(self, x):
#         h = torch.relu(self.fc1(x))
#         h = torch.relu(self.fc2(h))
#         return self.fc3_mean(h), self.fc3_log_var(h)

#     def reparameterize(self, mean, log_var):
#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn_like(std)
#         return mean + eps * std

#     def forward(self, x):
#         mean, log_var = self.encode(x)
#         z = self.reparameterize(mean, log_var)
#         return z, mean, log_var

# class DeepDEP(nn.Module):
#     def __init__(self, dims_mut, dims_exp, dims_cna, dims_meth, fprint_dim, dense_layer_dim):
#         super(DeepDEP, self).__init__()
#         self.vae_mut = VAE(dims_mut[0], [1000, 100], 50)
#         self.vae_exp = VAE(dims_exp[0], [500, 200], 50)
#         self.vae_cna = VAE(dims_cna[0], [500, 200], 50)
#         self.vae_meth = VAE(dims_meth[0], [500, 200], 50)

#         self.fc_gene1 = nn.Linear(fprint_dim, 1000)
#         self.fc_gene2 = nn.Linear(1000, 100)
#         self.fc_gene3 = nn.Linear(100, 50)

#         self.fc_merged1 = nn.Linear(250, dense_layer_dim)
#         self.fc_merged2 = nn.Linear(dense_layer_dim, dense_layer_dim)
#         self.fc_out = nn.Linear(dense_layer_dim, 1)

#     def forward(self, mut, exp, cna, meth, fprint):
#         z_mut, _, _ = self.vae_mut(mut)
#         z_exp, _, _ = self.vae_exp(exp)
#         z_cna, _, _ = self.vae_cna(cna)
#         z_meth, _, _ = self.vae_meth(meth)
        
#         gene = torch.relu(self.fc_gene1(fprint))
#         gene = torch.relu(self.fc_gene2(gene))
#         gene = torch.relu(self.fc_gene3(gene))
        
#         merged = torch.cat([z_mut, z_exp, z_cna, z_meth, gene], dim=1)
#         merged = torch.relu(self.fc_merged1(merged))
#         merged = torch.relu(self.fc_merged2(merged))
#         output = self.fc_out(merged)
#         return output

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
    plt.savefig(f'results/predictions/dependency_score_density_plot_{batch_size}_{learning_rate}_{epochs}_VAE_last.png')
    plt.show()

if __name__ == '__main__':
    model_name = "vae_model"  # "model_paper"
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
    model.load_state_dict(torch.load(f"results/models/{model_name}.pth", map_location=device))
    model.eval()

    # Load TCGA genomics data and gene fingerprints
    data_mut_tcga, data_labels_mut_tcga, sample_names_mut_tcga, gene_names_mut_tcga = load_data("Data/TCGA/tcga_mut_data_paired_with_ccl.txt")
    data_exp_tcga, data_labels_exp_tcga, sample_names_exp_tcga, gene_names_exp_tcga = load_data("Data/TCGA/tcga_exp_data_paired_with_ccl.txt")
    data_cna_tcga, data_labels_cna_tcga, sample_names_cna_tcga, gene_names_cna_tcga = load_data("Data/TCGA/tcga_cna_data_paired_with_ccl.txt")
    data_meth_tcga, data_labels_meth_tcga, sample_names_meth_tcga, gene_names_meth_tcga = load_data("Data/TCGA/tcga_meth_data_paired_with_ccl.txt")
    data_fprint_1298DepOIs, data_labels_fprint, gene_names_fprint, function_names_fprint = load_data("Data/crispr_gene_fingerprint_cgp.txt")
    print("\n\nDatasets successfully loaded.\n\n")

    batch_size = 512
    first_to_predict = 10
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

    y_true_train = np.loadtxt('results/predictions/y_true_train_CCL_VAE.txt', dtype=float)
    y_pred_train = np.loadtxt('results/predictions/y_pred_train_CCL_VAE.txt', dtype=float)

    plot_density(y_true_train[0:first_to_predict].flatten(),y_pred_train[0:first_to_predict].flatten(),data_pred.flatten(),5000,0.0001,10)

    # Write prediction results to txt
    data_pred_df = pd.DataFrame(data=np.transpose(data_pred), index=gene_names_fprint, columns=sample_names_mut_tcga[0:first_to_predict])
    data_pred_df.to_csv(f"results/predictions/tcga_predicted_data_{model_name}_demo.txt", sep='\t', index_label='CRISPR_GENE', float_format='%.4f')
    print("\n\nPrediction completed in %.1f mins.\nResults saved in /results/predictions/tcga_predicted_data_%s_demo.txt\n\n" % ((time.time()-t)/60, model_name))
