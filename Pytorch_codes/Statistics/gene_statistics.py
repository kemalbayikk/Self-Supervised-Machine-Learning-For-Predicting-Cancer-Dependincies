import pandas as pd

# Verilen verileri bir txt dosyasından okuyalım
file_path = 'results/predictions/tcga_predicted_data_deepdep_vae_model_9July.txt'

# Dosya içeriğini bir DataFrame'e yükleyelim
df = pd.read_csv(file_path, sep='\t')

# Ortalamaları ve standart sapmaları hesaplayalım
df.set_index('CRISPR_GENE', inplace=True)
df['mean'] = df.mean(axis=1)
df['std_deviation'] = df.std(axis=1)

# Sadece mean ve std_deviation sütunlarını içeren yeni bir DataFrame oluşturup CSV dosyasına yazalım
df_stats = df[['mean', 'std_deviation']]

output_file_stats = 'results/gene_statistics_only.csv'
df_stats.to_csv(output_file_stats)