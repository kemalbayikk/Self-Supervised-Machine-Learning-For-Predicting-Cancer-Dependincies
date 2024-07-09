import pandas as pd
import matplotlib.pyplot as plt
import os

# Veriyi yükle
data_path = 'results/predictions/tcga_predicted_data_deepdep_vae_model_9July.txt'  # Dosyanızın yolu
data = pd.read_csv(data_path, sep='\t')

cancer_type = "Breast_invasive_carcinoma"
# TCGA kodlarının hangi kanser türüne ait olduğunu gösteren dosyayı yükle
cancer_type_path = f'Data/CancerTCGAMappings/{cancer_type}.txt'  # Breast invasive kanser tipi dosyası
with open(cancer_type_path, 'r') as file:
    breast_invasive_tcga_ids = file.read().splitlines()

# Breast invasive kanser tipi ile ilgili kolonları filtrele
filtered_data = data[['CRISPR_GENE'] + breast_invasive_tcga_ids]

print(filtered_data)

# Genlerin ortalama dağılımını hesapla
filtered_data.set_index('CRISPR_GENE', inplace=True)
mean_distribution = filtered_data.mean(axis=1)

# En düşük 50 gene göre sırala
lowest_50_genes = mean_distribution.nsmallest(50)

# Grafiği oluştur ve kaydet
output_dir = 'results/tcga_gene_distribution_graphs'
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(10, 6))
lowest_50_genes.plot(kind='bar')
plt.title(f'Lowest 50 Genes in {cancer_type}')
plt.xlabel('Genes')
plt.ylabel('Mean Dependency Score')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'lowest_50_genes_{cancer_type}_9july.png'))
plt.close()