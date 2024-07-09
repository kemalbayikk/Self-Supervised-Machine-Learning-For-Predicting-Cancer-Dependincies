import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

gene_name = "EEF2"

# Kanser türlerini kısalt
cancer_type_short = {
    "Adrenocortical carcinoma": "Adrenal",
    "Bladder Urothelial Carcinoma": "Bladder",
    "Brain Lower Grade Glioma": "Brain",
    "Breast invasive carcinoma": "Breast",
    "Cervical squamous cell carcinoma and endocervical adenocarcinoma": "Cervical",
    "Cholangiocarcinoma": "Bile_Duct",
    "Colon adenocarcinoma": "Colon",
    "Esophageal carcinoma": "Esophageal",
    "Glioblastoma multiforme": "Brain",
    "Head and Neck squamous cell carcinoma": "Head_Neck",
    "Kidney Chromophobe": "Kidney",
    "Kidney renal clear cell carcinoma": "Kidney",
    "Kidney renal papillary cell carcinoma": "Kidney",
    "Liver hepatocellular carcinoma": "Liver",
    "Lung adenocarcinoma": "Lung",
    "Lung squamous cell carcinoma": "Lung",
    "Lymphoid Neoplasm Diffuse Large B-cell Lymphoma": "Lymphoid",
    "Mesothelioma": "Mesothelium",
    "Ovarian serous cystadenocarcinoma": "Ovarian",
    "Pancreatic adenocarcinoma": "Pancreatic",
    "Pheochromocytoma and Paraganglioma": "Adrenal",
    "Prostate adenocarcinoma": "Prostate",
    "Rectum adenocarcinoma": "Rectum",
    "Sarcoma": "Soft_Tissue",
    "Skin Cutaneous Melanoma": "Skin",
    "Stomach adenocarcinoma": "Stomach",
    "Testicular Germ Cell Tumors": "Testicular",
    "Thymoma": "Thymus",
    "Thyroid carcinoma": "Thyroid",
    "Uterine Carcinosarcoma": "Uterine",
    "Uterine Corpus Endometrial Carcinoma": "Uterine",
    "Uveal Melanoma": "Eye"
}

# Veriyi yükle
data_path = 'results/predictions/tcga_predicted_data_vae_model_last.txt'  # Verinizi yükledikten sonra dosya yolunu güncelleyin
data = pd.read_csv(data_path, sep='\t')

# TCGA kodlarının hangi kanser türüne ait olduğunu gösteren dosyayı yükle
tissue_source_file = 'Data/tissueSourceSite.tsv'
tissue_source_df = pd.read_csv(tissue_source_file, sep='\t')

# TSS kodları ile kanser türlerini eşleştir
tissue_source_df['TSS Code'] = tissue_source_df['TSS Code'].fillna('NA')
tss_to_cancer_type = tissue_source_df.set_index('TSS Code')['Study Name'].to_dict()

# CRISPR_GENE kolonunu index olarak ayarla
data.set_index('CRISPR_GENE', inplace=True)

# TCGA ID'lerini kanser türlerine dönüştür
cancer_types = []
for tcga_id in data.columns:
    tss_code = tcga_id.split('-')[1]
    cancer_type = tss_to_cancer_type.get(tss_code, 'Unknown').strip()
    cancer_types.append(cancer_type_short.get(cancer_type, cancer_type))

# DataFrame'in kolon adlarını kanser türleri ile değiştir
data.columns = cancer_types

# Belirtilen genin verilerini çek
gene_data = data.loc[gene_name]

# Box plot ve violin plot oluşturma
output_dir_box = 'results/gene_tcga_distribution_graphs/box_plots'
output_dir_violin = 'results/gene_tcga_distribution_graphs/violin_plots'
os.makedirs(output_dir_box, exist_ok=True)
os.makedirs(output_dir_violin, exist_ok=True)

# Box plot
plt.figure(figsize=(12, 8))
sns.boxplot(x=gene_data.index, y=gene_data.values)
plt.title(f'{gene_name} Dependency Scores Across Cancer Types (Box Plot)')
plt.xlabel('Cancer Types')
plt.ylabel('Dependency Score')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(output_dir_box, f'{gene_name}_dependency_scores_boxplot.png'))
plt.close()

# Violin plot
plt.figure(figsize=(12, 8))
sns.violinplot(x=gene_data.index, y=gene_data.values)
plt.title(f'{gene_name} Dependency Scores Across Cancer Types (Violin Plot)')
plt.xlabel('Cancer Types')
plt.ylabel('Dependency Score')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(output_dir_violin, f'{gene_name}_dependency_scores_violinplot.png'))
plt.close()

#print(f"Box plot ve violin plot grafikleri {output_dir} klasörüne kaydedildi.")
