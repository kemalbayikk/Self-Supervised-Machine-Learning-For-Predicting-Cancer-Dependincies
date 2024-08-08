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

        merged = torch.cat([mu_mut, mu_fprint], dim=1)
        merged = torch.relu(self.fc_merged1(merged))
        merged = torch.relu(self.fc_merged2(merged))
        output = self.fc_out(merged)
        return output
    
# import torch
# import torch.nn as nn
# import numpy as np
# import pandas as pd
# import time

# # VAE ve DeepDEP sınıfları kodun önceki kısmındaki gibi tanımlıdır.

# # Veri yükleme fonksiyonu
# def load_data(filename):
#     data = []
#     gene_names = []
#     data_labels = []
#     lines = open(filename).readlines()
#     sample_names = lines[0].replace('\n', '').split('\t')[1:]
#     dx = 1

#     for line in lines[dx:]:
#         values = line.replace('\n', '').split('\t')
#         gene = str.upper(values[0])
#         gene_names.append(gene)
#         data.append(values[1:])
#     data = np.array(data, dtype='float32')
#     data = np.transpose(data)

#     return data, data_labels, sample_names, gene_names

# if __name__ == '__main__':
#     device = "mps"
    
#     # Model boyutlarını tanımla ve modeli yükle
#     dims_mut = 4539
#     fprint_dim = 3115
#     dense_layer_dim = 100

#     model = DeepDEP(dims_mut, fprint_dim, dense_layer_dim).to(device)

#     model.load_state_dict(torch.load(f"PytorchStaticSplits/DeepDepMutationOnly/Results/Split2/PredictionNetworkModels/VAE_Prediction_Network_Split_2_Only_Mutation.pth", map_location=device))
#     model.eval()

#     # TCGA genomik verilerini ve gen fingerprint verilerini yükle
#     data_mut_tcga, data_labels_mut_tcga, sample_names_mut_tcga, gene_names_mut_tcga = load_data("Data/TCGA/tcga_mut_data_paired_with_ccl.txt")
#     data_fprint_1298DepOIs, data_labels_fprint, gene_names_fprint, function_names_fprint = load_data("Data/crispr_gene_fingerprint_cgp.txt")
#     print("\n\nDatasets successfully loaded.\n\n")

#     # BRCA1 ve TP53 için analiz yap
#     brca1_index = gene_names_mut_tcga.index('BRCA1')
#     tp53_index = gene_names_mut_tcga.index('TP53')
#     data_pred = np.zeros((len(sample_names_mut_tcga), data_fprint_1298DepOIs.shape[0]))
    
#     t = time.time()
#     for z in np.arange(0, 1):
#         # Mutasyon durumları için veri hazırlama
#         data_mut_batch = np.zeros((data_fprint_1298DepOIs.shape[0], dims_mut), dtype='float32')
#         data_mut_batch[:, :] = data_mut_tcga[z, :]  # Diğer genlerin mutasyon durumunu koru
#         data_mut_batch = torch.tensor(data_mut_batch, dtype=torch.float32).to(device)

#         data_fprint_batch = torch.tensor(data_fprint_1298DepOIs, dtype=torch.float32).to(device)

#         with torch.no_grad():
#             # 1. Hem BRCA1 hem de TP53 mutasyonlu
#             data_mut_batch[:, brca1_index] = 1.0
#             data_mut_batch[:, tp53_index] = 1.0
#             output_both_mut = model(data_mut_batch, data_fprint_batch).cpu().numpy()

#             # 2. Sadece BRCA1 mutasyonlu
#             data_mut_batch[:, brca1_index] = 1.0
#             data_mut_batch[:, tp53_index] = 0.0
#             output_brca1_only = model(data_mut_batch, data_fprint_batch).cpu().numpy()

#             # 3. Sadece TP53 mutasyonlu
#             data_mut_batch[:, brca1_index] = 0.0
#             data_mut_batch[:, tp53_index] = 1.0
#             output_tp53_only = model(data_mut_batch, data_fprint_batch).cpu().numpy()

#             # 4. Ne BRCA1 ne de TP53 mutasyonlu
#             data_mut_batch[:, brca1_index] = 0.0
#             data_mut_batch[:, tp53_index] = 0.0
#             output_none_mut = model(data_mut_batch, data_fprint_batch).cpu().numpy()
        
#         # SE skorları hesaplama
#         se_both_vs_none = output_both_mut - output_none_mut
#         se_brca1_vs_none = output_brca1_only - output_none_mut
#         se_tp53_vs_none = output_tp53_only - output_none_mut
#         se_score = se_both_vs_none - (se_brca1_vs_none + se_tp53_vs_none)
        
#         data_pred[z] = np.transpose(se_score)
#         print("TCGA sample %d predicted..." % z)

#     # Sonuçları txt dosyasına yaz
#     data_pred_df = pd.DataFrame(data=np.transpose(data_pred), index=gene_names_fprint, columns=sample_names_mut_tcga)
#     data_pred_df.to_csv(f"PytorchStaticSplits/SyntheticEssentiality/BRCA1_TP53_mut_predictions.txt", sep='\t', index_label='DepOI', float_format='%.4f')
#     print("\n\nPrediction completed in %.1f mins.\nResults saved in PytorchStaticSplits/SyntheticEssentiality/BRCA1_TP53_mut_predictions.txt\n\n" % ((time.time()-t)/60))


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
    device = "mps"

    gene = "BRCA1"
    
    # Model boyutlarını tanımla ve modeli yükle

    fprint_dim = 3115
    dense_layer_dim = 100

    # TCGA genomik verilerini ve gen fingerprint verilerini yükle
    data_mut_tcga, data_labels_mut_tcga, sample_names_mut_tcga, gene_names_mut_tcga = load_data("Data/CCL/ccl_mut_data_paired_with_tcga.txt") # load_data("Data/TCGA/tcga_mut_data_paired_with_ccl.txt")
    data_fprint_1298DepOIs, data_labels_fprint, gene_names_fprint, function_names_fprint = load_data("Data/crispr_gene_fingerprint_cgp.txt")
    print("\n\nDatasets successfully loaded.\n\n")

    dims_mut = 4539

    model = DeepDEP(dims_mut, fprint_dim, dense_layer_dim).to(device)
    model.load_state_dict(torch.load(f"PytorchStaticSplits/DeepDepMutationOnly/Results/Split2/PredictionNetworkModels/VAE_Prediction_Network_Split_2_Only_Mutation.pth", map_location=device))
    model.eval()

    # BRCA1 için analiz yap
    brca1_index = gene_names_mut_tcga.index(gene)
    data_pred = np.zeros((len(sample_names_mut_tcga), data_fprint_1298DepOIs.shape[0]))
    max_se_scores = []


    t = time.time()
    for z in np.arange(0, len(sample_names_mut_tcga)):
        # BRCA1 geninin mutasyon durumu (mutasyonlu ve mutasyonsuz)
        data_mut_batch = np.zeros((data_fprint_1298DepOIs.shape[0], dims_mut), dtype='float32')
        data_mut_batch[:, :] = data_mut_tcga[z, :]  # Diğer genlerin mutasyon durumunu koru
        data_mut_batch = torch.tensor(data_mut_batch, dtype=torch.float32).to(device)

        data_fprint_batch = torch.tensor(data_fprint_1298DepOIs, dtype=torch.float32).to(device)

        with torch.no_grad():
            # Mutasyonlu durum (BRCA1 geninde mutasyon var)
            data_mut_batch[:, brca1_index] = 1.0
            output_mut = model(data_mut_batch, data_fprint_batch).cpu().numpy()

            # Mutasyon olmayan durum (BRCA1 geninde mutasyon yok)
            data_mut_batch[:, brca1_index] = 0.0
            output_wt = model(data_mut_batch, data_fprint_batch).cpu().numpy()
        
        # SE skorları: mutasyonlu - mutasyonsuz
        se_scores = output_mut - output_wt
        data_pred[z] = np.transpose(se_scores)
        print("TCGA sample %d predicted..." % z)

        # Her TCGA örneği için en düşük SE skoruna sahip geni bul
        min_se_index = np.argmin(se_scores)
        max_se_scores.append((sample_names_mut_tcga[z], gene_names_fprint[min_se_index], se_scores[min_se_index][0]))

    # Sonuçları CSV dosyasına yaz
    data_pred_df = pd.DataFrame(data=np.transpose(data_pred), index=gene_names_fprint, columns=sample_names_mut_tcga)
    data_pred_df.to_csv(f"PytorchStaticSplits/SyntheticEssentiality/{gene}_only_mut_predictions_CCL.csv", index_label='DepOI', float_format='%.4f')

    # Her TCGA örneği için en yüksek SE skoruna sahip gen ve SE skorunu kaydet (küçükten büyüğe sıralı)
    max_se_scores_sorted = sorted(max_se_scores, key=lambda x: x[2])
    max_se_scores_df = pd.DataFrame(max_se_scores_sorted, columns=['Sample', 'Gene', 'SE_Score'])
    max_se_scores_df.to_csv(f"PytorchStaticSplits/SyntheticEssentiality/Highest_SE_Scores_{gene}_CCL.csv", index=False)

    print(f"\n\nPrediction completed in %.1f mins.\nResults saved in PytorchStaticSplits/SyntheticEssentiality/{gene}_only_mut_predictions_CCL.csv\n" % ((time.time()-t)/60))
    print(f"Highest SE scores saved in PytorchStaticSplits/SyntheticEssentiality/Highest_SE_Scores_{gene}_CCL.csv\n")
