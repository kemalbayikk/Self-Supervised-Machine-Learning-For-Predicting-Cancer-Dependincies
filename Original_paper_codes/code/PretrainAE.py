# Pretrain an autoencoder (AE) of tumor genomics (TCGA) to be used to initialize DeepDEP model training
print("\n\nStarting to run PretrainAE.py with a demo example of gene mutation data of 50 TCGA tumors...")

import pickle
from keras import models
from keras.layers import Dense, Merge
from keras.callbacks import EarlyStopping
import numpy as np
import time

def load_data(filename):
    data = []
    gene_names = []
    data_labels = []
    lines = open(filename).readlines()
    sample_names = lines[0].replace('\n', '').split('\t')[1:]
    dx = 1

    for line in lines[dx:]:
        values = line.replace('\n', '').split('\t')
        gene = str.upper(values[0])
        gene_names.append(gene)
        data.append(values[1:])
    data = np.array(data, dtype='float32')
    data = np.transpose(data)

    return data, data_labels, sample_names, gene_names

def AE_dense_3layers(input_dim, first_layer_dim, second_layer_dim, third_layer_dim, activation_func, init='he_uniform'):
    print('input_dim = ', input_dim)
    print('first_layer_dim = ', first_layer_dim)
    print('second_layer_dim = ', second_layer_dim)
    print('third_layer_dim = ', third_layer_dim)
    print('init = ', init)
    model = models.Sequential()
    model.add(Dense(output_dim = first_layer_dim, input_dim = input_dim, activation = activation_func, init = init))
    model.add(Dense(output_dim = second_layer_dim, input_dim = first_layer_dim, activation = activation_func, init = init))
    model.add(Dense(output_dim = third_layer_dim, input_dim = second_layer_dim, activation = activation_func, init = init))
    model.add(Dense(output_dim = second_layer_dim, input_dim = third_layer_dim, activation = activation_func, init = init))
    model.add(Dense(output_dim = first_layer_dim, input_dim = second_layer_dim, activation = activation_func, init = init))
    model.add(Dense(output_dim = input_dim, input_dim = first_layer_dim, activation = activation_func, init = init))
    
    return model

def save_weight_to_pickle(model, file_name):
    print('saving weights')
    weight_list = []
    for layer in model.layers:
        weight_list.append(layer.get_weights())
    with open(file_name, 'wb') as handle:
        pickle.dump(weight_list, handle)
        
if __name__ == '__main__':
    # load TCGA mutation data, substitute here with other genomics
    data_mut_tcga, data_labels_mut_tcga, sample_names_mut_tcga, gene_names_mut_tcga = load_data("/data/tcga_mut_data_paired_with_ccl.txt")
    print("\n\nDatasets successfully loaded.")
    
    samples_to_predict = np.arange(0, 50)
    # predict the first 50 samples for DEMO ONLY, for all samples please substitute 50 by data_mut_tcga.shape[0]
    # prediction results of all 8238 TCGA samples can be found in /data/premodel_tcga_*.pickle
    
    input_dim = data_mut_tcga.shape[1]
    first_layer_dim = 1000
    second_layer_dim = 100
    third_layer_dim = 50
    batch_size = 64
    epoch_size = 100
    activation_function = 'relu'
    init = 'he_uniform'
    model_save_name = "premodel_tcga_mut_%d_%d_%d" % (first_layer_dim, second_layer_dim, third_layer_dim)

    t = time.time()
    model = AE_dense_3layers(input_dim = input_dim, first_layer_dim = first_layer_dim, second_layer_dim=second_layer_dim, third_layer_dim=third_layer_dim, activation_func=activation_function, init=init)
    model.compile(loss = 'mse', optimizer = 'adam')
    model.fit(data_mut_tcga[samples_to_predict], data_mut_tcga[samples_to_predict], nb_epoch=epoch_size, batch_size=batch_size, shuffle=True)
    
    cost = model.evaluate(data_mut_tcga[samples_to_predict], data_mut_tcga[samples_to_predict], verbose = 0)
    print('\n\nAutoencoder training completed in %.1f mins.\n with testloss:%.4f' % ((time.time()-t)/60, cost))
    
    save_weight_to_pickle(model, '/results/autoencoders/' + model_save_name + '_demo.pickle')
    print("\nResults saved in /results/autoencoders/%s_demo.pickle\n\n" % model_save_name)
    