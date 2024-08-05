import pandas as pd

gene = "TP53"

# Load the Highest_SE_Scores_BRCA1.csv file
se_scores_df = pd.read_csv(F'PytorchStaticSplits/SyntheticEssentiality/Highest_SE_Scores_{gene}.csv')

# Load the tcga_cancer_types_mapped.txt file
cancer_types_df = pd.read_csv('Data/tcga_cancer_types_mapped.txt', sep='\t', header=None, names=['Sample', 'Cancer_Type'])

# Merge the dataframes on the 'Sample' column
merged_df = pd.merge(se_scores_df, cancer_types_df, left_on='Sample', right_on='Sample', how='inner')

# Group by 'Cancer_Type' and find the gene with the minimum mean SE score for each group
grouped_df = merged_df.groupby('Cancer_Type').apply(lambda x: x.loc[x['SE_Score'].idxmin()])

# Prepare the data for plotting
plot_data = grouped_df[['Cancer_Type', 'Gene', 'SE_Score']]
plot_data.set_index('Cancer_Type', inplace=True)

# Display the resulting DataFrame
# import ace_tools as tools; tools.display_dataframe_to_user(name="Grouped Cancer Types with Lowest SE Scores", dataframe=plot_data)

# Plot the data
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 8))
plot_data['SE_Score'].plot(kind='barh')
plt.xlabel('Minimum SE Score')
plt.ylabel('Cancer Type')
plt.title('Cancer Types with Genes Having the Lowest Mean SE Scores')
plt.gca().invert_yaxis()
plt.grid(axis='x')
plt.show()