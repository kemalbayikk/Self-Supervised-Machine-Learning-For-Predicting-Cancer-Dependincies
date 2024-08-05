import matplotlib.pyplot as plt
import pandas as pd

gene = "BRCA1"

# Load the Highest_SE_Scores_BRCA1.csv file
se_scores_df = pd.read_csv(F'PytorchStaticSplits/SyntheticEssentiality/Highest_SE_Scores_{gene}_CCL.csv')

# SE skorlarına göre sırala ve yalnızca en düşük 10 unique geni al
df_unique = se_scores_df.drop_duplicates(subset='Gene')
df_filtered = df_unique[df_unique['Gene'] != gene] 
lowest_scores_df = df_filtered.sort_values(by='SE_Score').head(20)

# Çizim
plt.figure(figsize=(10, 6))
plt.barh(lowest_scores_df['Gene'], lowest_scores_df['SE_Score'], color='skyblue')
plt.xlabel('SE Score')
plt.ylabel('Gene')
plt.title('Top 20 Most Synthetic Essential Genes with BRCA1 Mutation (Lowest SE Scores)')
plt.gca().invert_yaxis()  # En düşük skorlara göre yukarıdan aşağıya sıralama
plt.show()