import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

gene_name = 'HCC1395'
# Verileri yükleyelim
csv_data_path = 'Data/data_dep_updated_2.csv'  # CSV dosyası için düzgün yolu girin
txt_data_path = 'Pytorch_codes/Variational_autoencoder/Models To Analyze/Split 2 VAE/ccl_predicted_data_best_model_vae_split_2.txt'   # TXT dosyası için düzgün yolu girin

csv_data = pd.read_csv(csv_data_path)
txt_data = pd.read_csv(txt_data_path, delimiter='\t')  # Eğer dosya tab ile ayrılmışsa delimiter='\t' kullanın

## CRISPR_GENE sütununu koruyarak verileri geniş formatından uzun formata dönüştür
csv_melted = csv_data.melt(id_vars=['CRISPR_GENE'], var_name='CCL', value_name='Dependency Score').assign(Source='Original Data')
txt_melted = txt_data.melt(id_vars=['CRISPR_GENE'], var_name='CCL', value_name='Dependency Score').assign(Source='Predicted Data')

# CSV verilerindeki CCL'lerin ortalama etkisini hesapla
csv_ccl_average = csv_melted.groupby('CCL')['Dependency Score'].mean().reset_index()

# En düşük 10 CCL'yi seç
lowest_10_ccls_csv = csv_ccl_average.nsmallest(20, 'Dependency Score')['CCL']

# Top 10 CCL için orijinal verileri filtrele
filtered_csv_data = csv_melted[csv_melted['CCL'].isin(lowest_10_ccls_csv)]

# Aynı CCL'ler için tahmin edilen verileri filtrele
filtered_txt_data = txt_melted[txt_melted['CCL'].isin(lowest_10_ccls_csv)]

# Veri setlerini birleştir
combined_data = pd.concat([filtered_csv_data, filtered_txt_data])

# Violin plot çiz
# plt.figure(figsize=(30, 8))
# sns.violinplot(x='CCL', y='Dependency Score', hue='Source', data=combined_data, split=True, palette=['blue', 'orange'])
# plt.title('Comparison of Dependency Scores for 20 Highest Dependendent CCLs from Original and Predicted Data')
# plt.xticks(rotation=45)  # Daha fazla CCL ismini görmek için etiketleri döndür
# plt.xlabel('CCL')
# plt.ylabel('Dependency Score')
# plt.legend(title='Data Source', loc='upper right')

# Violin plot çiz
plt.figure(figsize=(30, 8))
sns.boxplot(x='CCL', y='Dependency Score', hue='Source', data=combined_data, palette=['blue', 'orange'])
plt.title('Comparison of Dependency Scores for 20 Highest Dependendent CCLs from Original and Predicted Data')
plt.xticks(rotation=45)  # Daha fazla CCL ismini görmek için etiketleri döndür
plt.xlabel('CCL')
plt.ylabel('Dependency Score')
plt.legend(title='Data Source', loc='upper right')


# Grafiği göster
plt.tight_layout()  # Layout'u düzenle
plt.show()