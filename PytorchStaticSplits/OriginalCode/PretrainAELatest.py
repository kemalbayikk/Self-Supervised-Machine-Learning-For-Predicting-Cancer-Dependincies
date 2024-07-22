import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os
import pickle
import time
from datetime import datetime
from keras import models
from keras.layers import Dense
from keras.callbacks import EarlyStopping

# Define the autoencoder model
def AE_dense_3layers(input_dim, first_layer_dim, second_layer_dim, third_layer_dim, activation_func, init='he_uniform'):
    model = models.Sequential()
    model.add(Dense(units=first_layer_dim, input_dim=input_dim, activation=activation_func, kernel_initializer=init))
    model.add(Dense(units=second_layer_dim, activation=activation_func, kernel_initializer=init))
    model.add(Dense(units=third_layer_dim, activation=activation_func, kernel_initializer=init))
    model.add(Dense(units=second_layer_dim, activation=activation_func, kernel_initializer=init))
    model.add(Dense(units=first_layer_dim, activation=activation_func, kernel_initializer=init))
    model.add(Dense(units=input_dim, activation=activation_func, kernel_initializer=init))
    return model

# Save weights to pickle file
def save_weight_to_pickle(model, file_name):
    weight_list = []
    for layer in model.layers:
        weight_list.append(layer.get_weights())
    with open(file_name, 'wb') as handle:
        pickle.dump(weight_list, handle)

# Load the pre-split datasets
def load_split_data(split_num, omic):
    base_path = f'PytorchStaticSplits/TCGASplits/split_{split_num}'
    train_file = os.path.join(base_path, f'train_dataset_{omic}_split_{split_num}.pth')
    val_file = os.path.join(base_path, f'val_dataset_{omic}_split_{split_num}.pth')
    test_file = os.path.join(base_path, f'test_dataset_{omic}_split_{split_num}.pth')
    
    train_dataset = torch.load(train_file)
    val_dataset = torch.load(val_file)
    test_dataset = torch.load(test_file)
    
    return train_dataset, val_dataset, test_dataset

# Main training script
if __name__ == '__main__':
    omic = "meth"
    
    for split_num in range(1, 6):
        train_dataset, val_dataset, test_dataset = load_split_data(split_num, omic)
        print(f"\nDatasets successfully loaded for split {split_num}.")
        
        train_data = np.array([data[0].numpy() for data in train_dataset])
        val_data = np.array([data[0].numpy() for data in val_dataset])
        test_data = np.array([data[0].numpy() for data in test_dataset])

        input_dim = train_data.shape[1]
        first_layer_dim = 500
        second_layer_dim = 200
        third_layer_dim = 50
        batch_size = 500
        epoch_size = 100
        activation_function = 'relu'
        init = 'he_uniform'
        model_save_name = f"premodel_tcga_mut_{first_layer_dim}_{second_layer_dim}_{third_layer_dim}_split_{split_num}"
        
        model = AE_dense_3layers(input_dim, first_layer_dim, second_layer_dim, third_layer_dim, activation_function, init)
        model.compile(loss='mse', optimizer='adam')
        
        t = time.time()
        model.fit(train_data, train_data, epochs=epoch_size, batch_size=batch_size, shuffle=True, validation_data=(val_data, val_data), callbacks=[EarlyStopping(patience=10)])
        
        cost = model.evaluate(test_data, test_data, verbose=0)
        print(f'\nAutoencoder training completed for split {split_num} in {(time.time()-t)/60:.1f} mins with test loss: {cost:.4f}')
        
        save_weight_to_pickle(model, f'PytorchStaticSplits/OriginalCode/Results/Split{split_num}/USL_Pretrained/tcga_{omic}_ae_best_split_{split_num}.pickle')
        print(f"PytorchStaticSplits/OriginalCode/Results/Split{split_num}/USL_Pretrained/tcga_{omic}_ae_best_split_{split_num}.pickle\n")
