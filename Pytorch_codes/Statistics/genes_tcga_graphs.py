import pandas as pd
import matplotlib.pyplot as plt
import os

gene_name = "RAN"

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

print(tss_to_cancer_type.keys())

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

# Her genin ortalama skorlarını hesapla
grouped_data = data.groupby(axis=1, level=0).mean()
print(grouped_data.loc[gene_name])

# Grafiği oluştur ve kaydet
output_dir = 'results/gene_tcga_distribution_graphs'
os.makedirs(output_dir, exist_ok=True)

# Her genin ortalama skorunu grafikleştir
# for gene in data.index:
plt.figure(figsize=(10, 6))
grouped_data.loc[gene_name].plot(kind='bar')
plt.title(f'{gene_name} Average Dependency Scores Across Cancer Types')
plt.xlabel('Cancer Types')
plt.ylabel('Dependency Score')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'{gene_name}_average_dependency_scores.png'))
plt.close()
