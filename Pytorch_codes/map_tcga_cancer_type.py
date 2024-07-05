# Load the uploaded TSV file and map the TCGA codes to cancer types
import pandas as pd

# Load the tissue source site codes file
file_path = 'Data/tissueSourceSite.tsv'
tissue_source_df = pd.read_csv(file_path, sep='\t')

# Display the first few rows to understand its structure
tissue_source_df.head()
# Create a dictionary to map TSS Code to Study Name
tss_to_study_name = tissue_source_df.set_index('TSS Code')['Study Name'].to_dict()

# Read the TCGA IDs from the file and map them to their corresponding cancer types
tcga_file_path = 'Data/tcgas.txt'
with open(tcga_file_path, 'r') as file:
    tcga_ids = file.read().split()

# Extract the project codes from the TCGA IDs and map them to cancer types
cancer_mapping = []
for tcga_id in tcga_ids:
    project_code = tcga_id.split('-')[1]  # Extract the project code from the TCGA ID
    cancer_type = tss_to_study_name.get(project_code, "Unknown")
    cancer_mapping.append(f"{tcga_id}\t{cancer_type}")

# Write the mapped data to a new text file
output_path = 'Data/tcga_cancer_types_mapped.txt'
with open(output_path, 'w') as output_file:
    output_file.write("\n".join(cancer_mapping))

from collections import defaultdict
import os

# Create a dictionary to store TCGA IDs by cancer type
cancer_type_dict = defaultdict(list)

# Populate the dictionary with TCGA IDs grouped by cancer type
for entry in cancer_mapping:
    tcga_id, cancer_type = entry.split('\t')
    cancer_type_dict[cancer_type].append(tcga_id)

# Base directory to save the files
base_dir = 'Data/CancerTCGAMappings'

# Ensure the base directory exists
os.makedirs(base_dir, exist_ok=True)

# Write the TCGA IDs to separate files based on their cancer type
output_files = {}
for cancer_type, ids in cancer_type_dict.items():
    # Create a valid filename from the cancer type
    filename = os.path.join(base_dir, f"{cancer_type.replace(' ', '_').replace('/', '_')}.txt")
    with open(filename, 'w') as output_file:
        output_file.write("\n".join(ids))
    output_files[cancer_type] = filename

