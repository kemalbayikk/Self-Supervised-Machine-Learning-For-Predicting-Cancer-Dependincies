# import pandas as pd
# import numpy as np

# # TCGA mutasyon verisi ve dependency skoru verisi yükleme

# mutation_data = pd.read_csv('Data/TCGA/tcga_mut_data_paired_with_ccl.txt', sep='\t')
# # mutation_data = pd.read_csv('Data/CCL/ccl_mut_data_paired_with_tcga.txt', sep='\t')


# dependency_scores = pd.read_csv('Pytorch_codes/Variational_autoencoder/Models To Analyze/Split 2 VAE/tcga_predicted_data_best_model_vae_split_2.txt', sep='\t')
# # dependency_scores = pd.read_csv('Pytorch_codes/Variational_autoencoder/Models To Analyze/Split 2 VAE/ccl_predicted_data_best_model_vae_split_2.txt', sep='\t')


# # Prediction veri setindeki genleri al
# predicted_genes = dependency_scores['CRISPR_GENE'].unique()

# # Mutasyon veri setinde bu genleri filtrele
# mutation_data_filtered = mutation_data[mutation_data['Gene'].isin(predicted_genes)]

# print(mutation_data_filtered)

# gene1 = 'PTEN'
# gene2 = 'PIK3CB'
# # Veri setlerini transpoze ederek genleri satırlara, TCGA ID'leri sütunlara al
# dependency_scores.set_index('CRISPR_GENE', inplace=True)
# dependency_scores = dependency_scores.transpose()

# mutation_data.set_index('Gene', inplace=True)
# mutation_data = mutation_data.transpose()

# # İki veri setini TCGA ID'lerine göre birleştir
# combined_data = dependency_scores.join(mutation_data, lsuffix='_dep', rsuffix='_mut')

# # Etkileşim etkisini hesaplama: örneğin, AARS2 ve A1CF için
# def interaction_effect(row, gene1, gene2):
#     # Sadece her iki gende de mutasyon varsa etkileşim hesaplanacak
#     if row[gene1 + '_mut'] > 0 and row[gene2 + '_mut'] > 0:
#         return row[gene1 + '_dep'] + row[gene2 + '_dep']
#     return np.nan

# # AARS2 ve A1CF için etkileşim etkisini hesapla
# combined_data[f'interaction_{gene1}_{gene2}'] = combined_data.apply(lambda row: interaction_effect(row, gene1, gene2), axis=1)

# # En düşük 5 etkileşim skoruna sahip TCGA örneklerini bul ve yazdır
# lowest_five = combined_data.nsmallest(5, f'interaction_{gene1}_{gene2}')

# print("En düşük etkileşim skoruna sahip 5 TCGA örneği:")
# print(lowest_five[[f'interaction_{gene1}_{gene2}']])

# # Etkileşim etkisini analiz etme
# interaction_results = combined_data[f'interaction_{gene1}_{gene2}'].dropna()

# print("AARS2 ve ABL1 genlerinde mutasyon olan örnekler için dependency skorlarının etkileşim etkisi:")
# print(interaction_results.describe())

import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm

# TCGA mutasyon verisi ve dependency skoru verisi yükleme
mutation_data = pd.read_csv('Data/TCGA/tcga_mut_data_paired_with_ccl.txt', sep='\t')
dependency_scores = pd.read_csv('Pytorch_codes/Variational_autoencoder/Models To Analyze/Split 2 VAE/tcga_predicted_data_best_model_vae_split_2.txt', sep='\t')

# Get genes from datasets
predicted_genes = set(dependency_scores['CRISPR_GENE'].unique())
mutation_genes = set(mutation_data['Gene'].unique())

# Find common genes between datasets
common_genes = predicted_genes.intersection(mutation_genes)

# Transpose datasets to make genes as rows and TCGA IDs as columns
dependency_scores.set_index('CRISPR_GENE', inplace=True)
dependency_scores = dependency_scores.transpose()

mutation_data.set_index('Gene', inplace=True)
mutation_data = mutation_data.transpose()

# Join datasets on TCGA IDs
combined_data = dependency_scores.join(mutation_data, lsuffix='_dep', rsuffix='_mut')

# Calculate interaction effects for all gene pairs
interactions = {}  # Dictionary to store interaction dataframes

print('Total combinations : ', len(list(itertools.combinations(common_genes, 2))))

for gene1, gene2 in tqdm(itertools.combinations(common_genes, 2), total=len(list(itertools.combinations(common_genes, 2))), desc="Calculating interactions"):
    def interaction_effect(row):
        if row[gene1 + '_mut'] > 0 and row[gene2 + '_mut'] > 0:
            return row[gene1 + '_dep'] + row[gene2 + '_dep']
        return np.nan

    # Apply the interaction effect function and store the result in a dictionary
    interactions[f'interaction_{gene1}_{gene2}'] = combined_data.apply(lambda row: interaction_effect(row), axis=1)

# Concatenate all interaction columns into the original DataFrame
interaction_df = pd.concat(interactions, axis=1)
combined_data = pd.concat([combined_data, interaction_df], axis=1)

# Now you can process `combined_data` to find the minimal scores and corresponding TCGA samples
results = []
for key, series in interaction_df.items():
    min_value = series.min()
    if pd.notna(min_value):
        min_tcga_samples = series[series == min_value].index.tolist()
        results.append((key, min_value, min_tcga_samples))

# Convert results to DataFrame and find top 10 results
results_df = pd.DataFrame(results, columns=['Gene_Pair', 'Min_Score', 'TCGA_Samples'])
top_results = results_df.nsmallest(10, 'Min_Score')

top_results_thoushand = results_df.nsmallest(1000, 'Min_Score')

# Save the top results to a CSV file
top_results_thoushand.to_csv('top_synthetic_lethality_results.csv')

# Print top results
print("Top 10 synthetic lethality scores with associated TCGA samples:")
print(top_results)
