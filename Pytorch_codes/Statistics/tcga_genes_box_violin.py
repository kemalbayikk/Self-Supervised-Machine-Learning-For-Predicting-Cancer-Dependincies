import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Veriyi yükle
data_path = 'results/predictions/tcga_predicted_data_vae_model_last.txt'  # Dosyanızın yolu
data = pd.read_csv(data_path, sep='\t')

cancer_type = "Breast_invasive_carcinoma"
# TCGA kodlarının hangi kanser türüne ait olduğunu gösteren dosyayı yükle
cancer_type_path = f'Data/CancerTCGAMappings/{cancer_type}.txt'  # Breast invasive kanser tipi dosyası
with open(cancer_type_path, 'r') as file:
    breast_invasive_tcga_ids = file.read().splitlines()

# Breast invasive kanser tipi ile ilgili kolonları filtrele
filtered_data = data[['CRISPR_GENE'] + breast_invasive_tcga_ids]

# Genlerin ortalama dağılımını hesapla
filtered_data.set_index('CRISPR_GENE', inplace=True)
mean_distribution = filtered_data.mean(axis=1)

# En düşük 50 gene göre sırala
lowest_50_genes = mean_distribution.nsmallest(30)

# En düşük 50 gene ait verileri al
lowest_50_genes_data = filtered_data.loc[lowest_50_genes.index]

# Veriyi uzun formata dönüştür
lowest_50_genes_data_long = lowest_50_genes_data.reset_index().melt(id_vars='CRISPR_GENE', var_name='TCGA_ID', value_name='Dependency_Score')

# Box plot ve violin plot oluşturma
# Box plot ve violin plot oluşturma
output_dir_box = 'results/variational_autoencoders/tcga_gene_distribution_graphs/box_plots'
output_dir_violin = 'results/variational_autoencoders/tcga_gene_distribution_graphs/violin_plots'
os.makedirs(output_dir_box, exist_ok=True)
os.makedirs(output_dir_violin, exist_ok=True)

# Box plot
plt.figure(figsize=(12, 8))
sns.boxplot(x='CRISPR_GENE', y='Dependency_Score', data=lowest_50_genes_data_long)
plt.title(f'Lowest 30 Genes in {cancer_type} (Box Plot)')
plt.xlabel('Genes')
plt.ylabel('Dependency Score')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(output_dir_box, f'lowest_30_genes_{cancer_type}_boxplot.png'))
plt.close()

# Violin plot
plt.figure(figsize=(12, 8))
sns.violinplot(x='CRISPR_GENE', y='Dependency_Score', data=lowest_50_genes_data_long)
plt.title(f'Lowest 30 Genes in {cancer_type} (Violin Plot)')
plt.xlabel('Genes')
plt.ylabel('Dependency Score')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(output_dir_violin, f'lowest_30_genes_{cancer_type}_violinplot.png'))
plt.close()
