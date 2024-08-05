import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Verileri yükleyelim
csv_data_path = 'Data/data_dep_updated_2.csv'  # CSV dosyası için düzgün yolu girin
txt_data_path = 'PytorchStaticSplits/DeepDepVAE/Analysis/ccl_predicted_data.txt'  # TXT dosyası için düzgün yolu girin

csv_data = pd.read_csv(csv_data_path)
txt_data = pd.read_csv(txt_data_path, delimiter='\t')  # Eğer dosya tab ile ayrılmışsa delimiter='\t' kullanın

# CRISPR_GENE sütununu koruyarak verileri geniş formatından uzun formata dönüştür
csv_melted = csv_data.melt(id_vars=['CRISPR_GENE'], var_name='CCL', value_name='Dependency Score').assign(Source='Original Data')
txt_melted = txt_data.melt(id_vars=['CRISPR_GENE'], var_name='CCL', value_name='Dependency Score').assign(Source='Predicted Data')

# Her iki veri seti için genlerin ortalama etkisini hesapla
csv_average_impact = csv_melted.groupby('CRISPR_GENE')['Dependency Score'].mean().reset_index()

# En etkili 30 geni seç
top_30_genes_csv = csv_average_impact.nsmallest(20, 'Dependency Score')['CRISPR_GENE']

# Veri setlerini birleştir
filtered_csv_data = csv_melted[csv_melted['CRISPR_GENE'].isin(top_30_genes_csv)]
filtered_txt_data = txt_melted[txt_melted['CRISPR_GENE'].isin(top_30_genes_csv)]

combined_data = pd.concat([filtered_csv_data, filtered_txt_data])

gene_order = combined_data.groupby('CRISPR_GENE')['Dependency Score'].mean().sort_values().index

# # Violin plot çiz
plt.figure(figsize=(30, 8))
sns.violinplot(x='CRISPR_GENE', y='Dependency Score', hue='Source', data=combined_data, split=True, palette=['blue', 'orange'], order=gene_order)
plt.title('Comparison of Dependency Scores for Top 20 Most Effective Genes Across All CCLs')
plt.xticks(rotation=90)  # Daha fazla gen ismini görmek için etiketleri döndür
plt.xlabel('Gene')
plt.ylabel('Dependency Score')
plt.legend(title='Data Source', loc='upper right')

# Violin plot çiz
# plt.figure(figsize=(30, 8))
# sns.boxplot(x='CRISPR_GENE', y='Dependency Score', hue='Source', data=combined_data, palette=['blue', 'orange'], order=gene_order)
# plt.title('Comparison of Dependency Scores for Top 20 Most Effective Genes Across All CCLs')
# plt.xticks(rotation=90)  # Daha fazla gen ismini görmek için etiketleri döndür
# plt.xlabel('Gene')
# plt.ylabel('Dependency Score')
# plt.legend(title='Data Source', loc='upper right')

# Grafiği göster
plt.tight_layout()  # Layout'u düzenle
plt.show()