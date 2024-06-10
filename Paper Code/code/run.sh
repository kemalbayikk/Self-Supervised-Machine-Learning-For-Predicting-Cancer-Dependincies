# model training, validation, and testing
mkdir -p /results/models/
python3.5 TrainNewModel.py

# predict TCGA (or other new) samples using a trained model
mkdir -p /results/predictions/
python3.5 PredictNewSamples.py

# pretrain an autoencoder (AE) of tumor genomics using the TCGA tumor samples that will be used to initialize DeepDEP model training
mkdir -p /results/autoencoders/
python3.5 PretrainAE.py