import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

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

        merged = torch.cat([mu_mut, mu_fprint], dim=1)
        merged = torch.relu(self.fc_merged1(merged))
        merged = torch.relu(self.fc_merged2(merged))
        output = self.fc_out(merged)
        return output

# Veri yükleme fonksiyonu
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model boyutlarını tanımla ve modeli yükle
    dims_mut = 4539
    fprint_dim = 3115
    dense_layer_dim = 100

    model = DeepDEP(dims_mut, fprint_dim, dense_layer_dim).to(device)
    model.load_state_dict(torch.load(f"PytorchStaticSplits/DeepDepMutationOnly/Results/Split2/PredictionNetworkModels/VAE_Prediction_Network_Split_2_Only_Mutation.pth", map_location=device))
    model.eval()

    # TCGA genomik verilerini ve gen fingerprint verilerini yükle
    data_mut_tcga, data_labels_mut_tcga, sample_names_mut_tcga, gene_names_mut_tcga = load_data("Data/CCL/ccl_mut_data_paired_with_tcga.txt")
    data_fprint_1298DepOIs, data_labels_fprint, gene_names_fprint, function_names_fprint = load_data("Data/crispr_gene_fingerprint_cgp.txt")
    print("\n\nDatasets successfully loaded.\n\n")

    # BRCA1 için indeks al
    brca1_index = gene_names_mut_tcga.index('BRCA1')
    data_pred = np.zeros((len(sample_names_mut_tcga), data_fprint_1298DepOIs.shape[0]))

    t = time.time()
    for z in range(len(sample_names_mut_tcga)):
        for partner_gene_index, partner_gene_name in enumerate(gene_names_mut_tcga):
            if partner_gene_name == 'BRCA1':
                continue  # Kendisiyle karşılaştırmayı atla

            data_mut_batch = np.zeros((2, dims_mut), dtype='float32')
            data_mut_batch[:, :] = data_mut_tcga[z, :]  # Diğer genlerin mutasyon durumunu koru

            # Her örnek için BRCA1 ve partner gen kombinasyonları
            data_mut_batch[0, brca1_index] = 1.0  # BRCA1 mutant
            data_mut_batch[1, brca1_index] = 0.0  # BRCA1 wildtype
            data_mut_batch[0, partner_gene_index] = 1.0  # Partner gen mutant
            data_mut_batch[1, partner_gene_index] = 0.0  # Partner gen wildtype

            data_mut_batch = torch.tensor(data_mut_batch, dtype=torch.float32).to(device)
            data_fprint_batch = torch.tensor(data_fprint_1298DepOIs, dtype=torch.float32).to(device)

            with torch.no_grad():
                output = model(data_mut_batch, data_fprint_batch).cpu().numpy()
            
            # SE skoru: (BRCA1 mutant + partner gen mutant) - (BRCA1 wildtype + partner gen wildtype)
            se_scores = output[0] - output[1]
            data_pred[z] += se_scores

            print(f"Sample {z}, Partner Gene {partner_gene_name} analyzed...")

    # Sonuçları CSV dosyasına yaz
    data_pred_df = pd.DataFrame(data=np.transpose(data_pred), index=gene_names_fprint, columns=sample_names_mut_tcga)
    data_pred_df.to_csv(f"PytorchStaticSplits/SyntheticLethality/BRCA1_mut_syn_leth_predictions_CCL.csv", index_label='DepOI', float_format='%.4f')

    # Her gen için ortalama SE skorunu hesaplama
    average_se_scores = data_pred_df.mean(axis=1)

    # En düşük ortalama SE skorlarına sahip 20 genin seçimi
    lowest_se_genes = average_se_scores.nsmallest(20)

    # Grafiği çizme
    plt.figure(figsize=(10, 6))
    plt.barh(lowest_se_genes.index, lowest_se_genes.values, color='skyblue')
    plt.xlabel('Average SE Score')
    plt.ylabel('Genes')
    plt.title('Top 20 Most Synthetic Lethal Genes with BRCA1 Mutation (Lowest SE Scores)')
    plt.gca().invert_yaxis()
    plt.grid(axis='x')

    plt.show()

    print(f"\n\nPrediction completed in {(time.time()-t)/60:.1f} mins.\nResults saved in PytorchStaticSplits/SyntheticLethality/BRCA1_mut_syn_leth_predictions_CCL.csv\n")
