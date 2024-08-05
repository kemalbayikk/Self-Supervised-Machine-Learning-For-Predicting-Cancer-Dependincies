import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time

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
    def __init__(self, dims_mut, fprint_dim, dense_layer_dim):
        super(DeepDEP, self).__init__()
        self.vae_mut = VariationalAutoencoder(dims_mut, 1000, 100, 50)
        self.vae_fprint = VariationalAutoencoder(fprint_dim, 1000, 100, 50)

        self.fc_merged1 = nn.Linear(dense_layer_dim, dense_layer_dim)
        self.fc_merged2 = nn.Linear(dense_layer_dim, dense_layer_dim)
        self.fc_out = nn.Linear(dense_layer_dim, 1)

    def forward(self, mut, fprint):
        recon_mut, mu_mut, logvar_mut = self.vae_mut(mut)
        recon_fprint, mu_fprint, logvar_fprint = self.vae_fprint(fprint)

        print("mu_mut size:", mu_mut.size())
        print("mu_fprint size:", mu_fprint.size())
        
        merged = torch.cat([mu_mut, mu_fprint], dim=1)
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

if __name__ == '__main__':
    device = "mps"
    
    # Define the model architecture with correct dimensions
    dims_mut = 4539  # Correct dimension based on the error message
    dims_exp = 6016
    dims_cna = 7460
    dims_meth = 6617
    fprint_dim = 3115  # Correct dimension based on the error message
    dense_layer_dim = 100

    model = DeepDEP(dims_mut, fprint_dim, dense_layer_dim).to(device)

    # Load the PyTorch model state dictionary
    model.load_state_dict(torch.load(f"PytorchStaticSplits/DeepDepMutationOnly/Results/Split2/PredictionNetworkModels/VAE_Prediction_Network_Split_2_Only_Mutation.pth", map_location=device))
    model.eval()

    # Load TCGA genomics data and gene fingerprints
    data_mut_tcga, data_labels_mut_tcga, sample_names_mut_tcga, gene_names_mut_tcga = load_data("Data/TCGA/tcga_mut_data_paired_with_ccl.txt")
    data_fprint_1298DepOIs, data_labels_fprint, gene_names_fprint, function_names_fprint = load_data("Data/crispr_gene_fingerprint_cgp.txt")
    print("\n\nDatasets successfully loaded.\n\n")

    batch_size = 10000
    first_to_predict = 8238
    data_pred = np.zeros((first_to_predict, data_fprint_1298DepOIs.shape[0]))
    
    t = time.time()
    for z in np.arange(0, first_to_predict):
        data_mut_batch = torch.tensor(data_mut_tcga[np.repeat(z, data_fprint_1298DepOIs.shape[0])], dtype=torch.float32).to(device)
        data_fprint_batch = torch.tensor(data_fprint_1298DepOIs, dtype=torch.float32).to(device)

        with torch.no_grad():
            data_pred_tmp = model(data_mut_batch, data_fprint_batch).cpu().numpy()
        
        data_pred[z] = np.transpose(data_pred_tmp)
        print("TCGA sample %d predicted..." % z)

    # Write prediction results to txt
    data_pred_df = pd.DataFrame(data=np.transpose(data_pred), index=gene_names_fprint, columns=sample_names_mut_tcga[0:first_to_predict])
    data_pred_df.to_csv(f"PytorchStaticSplits/SyntheticEssentiality/only_mut_predictions.txt", sep='\t', index_label='DepOI', float_format='%.4f')
    print("\n\nPrediction completed in %.1f mins.\nResults saved in PytorchStaticSplits/SyntheticEssentiality/only_mut_predictions.txt\n\n" % ((time.time()-t)/60))
