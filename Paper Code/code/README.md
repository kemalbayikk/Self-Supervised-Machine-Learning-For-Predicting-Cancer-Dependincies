# DeepDEP: Deep Learning of a Cancer Dependency Map Using Cancer Genomics

<img align="center" src="https://github.com/chenlabgccri/Prep4DeepDEP/blob/master/sketch/DeepDEP/DeepDEP_model_design.png?raw=true" alt="drawing" width="800">

## Introduction
This is a Python implementation of DeepDEP. DeepDEP is a deep learning model that predicts the gene dependency profile of an unscreened cancer cell line (CCL) or impracticable-to-screen tumors based on the baseline genomic profiles.

## Getting Started
**Deep learning architecture:**
- Dimension-reducing encoder neural networks for each type of molecular data, including DNA mutation, gene expression, DNA methylation, and copy number alteration (CNA).
- An encoder network for abstracting functional fingerprints of a gene dependency of interest (DepOI).
- A prediction network to convert the learned features into a dependency (gene-effect) score.

**The package has three main Python programs that:**
- Train a DeepDEP model with user's in-house single- or multi-omic profiles and gene dependency data.
- Predict gene dependencies of new samples (e.g., tumors) based on single or multi-omic profiles and a pretrained DeepDEP model.
- Pretrain an autoencoder (AE) of each genomic profile of tumors to be used to initialize DeepDEP training.

## Using the Codes
**View the results of our Reproducible Run:**
- Check the right panel for results from a Reproducible Run.

**Run user's own data on Code Ocean:**
- 'Capsule' -> 'Duplicate' to duplicate the capsule to user's own dashboard.

**Download the capsule to user's local machine:**
- 'Capsule' -> 'Export...' to download a zip file.
- 'Capsule' -> 'Clone via Git...' to git clone.

## Codes Details
The /code directory contains all major codes of DeepDEP. Here are brief descriptions:

#### *TrainNewModel.py*
Code to train, validate, and test single-, 2-, and full 4-omics DeepDEP models.

*Inputs:*

- *.txt* file for each genomic data (dimension: #genomic features by #CCL-DepOI pairs)
- *.txt* file for gene fingerprints (dimension: #fingerprint features [gene sets] by #CCL-DepOIs)
- Examples of the genomic data and gene fingerprints: DEMO dataset of 28 CCLs x 1298 DepOIs = 36344 samples is available at /data/ccl_complete_data_28CCL_1298DepOI_36344samples_demo.pickle. The complete data used in the paper [278 CCLs x 1298 DepOIs = 360844 samples; ~37 GB] can be downloaded [here](https://drive.google.com/file/d/1kK1XxgqgYmXrJJWqd27zz1bmUyHWFeSn/view?usp=sharing). The datasets can be prepared using our R package [*Prep4DeepDEP*](https://github.com/ChenLabGCCRI/Prep4DeepDEP/).
- *.pickle* file of an autoencoder for each genomic data to initialize the DeepDEP model. TCGA results used for the paper: /data/premodel_tcga_\*.pickle. Autoencoders can be obtained from *PretrainAE.py*.

*Output:*

- *.h5* file of the DeepDEP model.
DeepDEP models of single-, 2-, and full 4-omics of the paper: /data/full_results_models_paper/models/ and DEMO output: /results/models/

#### *PredictNewSamples.py*
Code to predict TCGA (or other new) samples using a trained DeepDEP model. Users may feed genomics profiles into an appropriate model (with single-, 2-, and full 4-omics) and predict dependency profiles without additional training.

*Inputs:*
- *.h5* file of a DeepDEP model to be used for prediction. Results reported in the paper: /data/full_results_models_paper/models/\*.h5
- *.txt* file for each genomic data (dimension: #genomic features by #CCLs/tumors). TCGA data used in the paper: /data/tcga_\*_data_paired_with_ccl.txt and CCL data used in the paper: /data/ccl_\*_data_paired_with_tcga.txt. The datasets can be prepared using our R package [*Prep4DeepDEP*](https://github.com/ChenLabGCCRI/Prep4DeepDEP/).
- .txt file for gene fingerprints (dimension: #fingerprint features [gene sets] by #DepOIs). Data used in the paper (1298 DepOIs): /data/crispr_gene_fingerprint_cgp.txt

*Output:*

- *.txt* file of predicted dependency scores (dimension: #DepOIs by #samples).
Results reported in the paper of the TCGA tumors: /data/full_results_models_paper/predictions/tcga_predicted_data_model_paper.txt, CCL using 10-fold cross-validations: /data/full_results_models_paper/predictions/ccl_predicted_data_model_10xCV_paper.txt, and DEMO output (10 tumors sampled from TCGA): /results/predictions/

#### *PretrainAE.py*
Code to pretrain an autoencoder (AE) of tumor genomics to be used to initialize DeepDEP training.

*Input:*

- *.txt* file of genomic data of samples (e.g., tumors of TCGA). Dimension: #genomic features by #CCLs/tumors. TCGA data used in the paper: /data/tcga_\*_data_paired_with_ccl.txt

*Output:*

- *.pickle* file of an autoencoder for each genomic data. TCGA results used for the paper: /data/premodel_tcga_\*.pickle and DEMO results (50 tumors sampled from TCGA): /results/autoencoders/

#### Flowchart:
<img align="center" src="https://github.com/chenlabgccri/Prep4DeepDEP/blob/master/sketch/DeepDEP/DeepDEP_codes_usage.png?raw=true" alt="drawing" width="800">

## Accompanying R Package to Prepare Model Inputs
The *Prep4DeepDEP* R package is provided at [GitHub](https://github.com/ChenLabGCCRI/Prep4DeepDEP/) to prepare the genomic features and generate the gene fingerprints to use the pretrained DeepDEP models that we provide here.

## Reference
Chiu YC, Zheng S, Wang LJ, Iskra BS, Rao MK, Houghton PJ, Huang Y, Chen Y.
"Predicting and characterizing a cancer dependency map of tumors with deep learning."  <em>Science Advances</em>. 2021.

## Contact Information
**Yu-Chiao Chiu:** ChiuY at uthscsa.edu; **Yufei Huang:** Yufei.Huang at utsa.edu; **Yidong Chen:** ChenY8 at uthscsa.edu
