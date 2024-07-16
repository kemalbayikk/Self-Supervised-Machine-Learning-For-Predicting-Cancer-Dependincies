# Train, validate, and test single-, 2-, and full 4-omics DeepDEP models
print("\n\nStarting to run TrainNewModel.py with a demo example of 28 CCLs x 1298 DepOIs...")

import pickle
from keras import models
from keras.layers import Dense, Merge
from keras.callbacks import EarlyStopping
import numpy as np
import time

if __name__ == '__main__':
    with open('/data/ccl_complete_data_28CCL_1298DepOI_36344samples_demo.pickle', 'rb') as f:
        data_mut, data_exp, data_cna, data_meth, data_dep, data_fprint = pickle.load(f)
    # This pickle file is for DEMO ONLY (containing 28 CCLs x 1298 DepOIs = 36344 samples)!
    # First 1298 samples correspond to 1298 DepOIs of the first CCL, and so on.
    # For the complete data used in the paper (278 CCLs x 1298 DepOIs = 360844 samples),
    # please substitute by 'ccl_complete_data_278CCL_1298DepOI_360844samples.pickle',
    # to which a link can be found in README.md
    
    # Load autoencoders of each genomics that were pre-trained using 8238 TCGA samples
    # New autoencoders can be pretrained using PretrainAE.py
    premodel_mut = pickle.load(open('/data/premodel_tcga_mut_1000_100_50.pickle', 'rb'))
    premodel_exp = pickle.load(open('/data/premodel_tcga_exp_500_200_50.pickle', 'rb'))
    premodel_cna = pickle.load(open('/data/premodel_tcga_cna_500_200_50.pickle', 'rb'))
    premodel_meth = pickle.load(open('/data/premodel_tcga_meth_500_200_50.pickle', 'rb'))
    print("\n\nDatasets successfully loaded.")

    activation_func = 'relu' # for all middle layers
    activation_func2 = 'linear' # for output layer to output unbounded gene-effect scores
    init = 'he_uniform'
    dense_layer_dim = 250
    batch_size = 500
    num_epoch = 100
    num_DepOI = 1298 # 1298 DepOIs as defined in our paper
    num_ccl = int(data_mut.shape[0]/num_DepOI)
    
    # 90% CCLs for training/validation, and 10% for testing
    id_rand = np.random.permutation(num_ccl)
    id_cell_train = id_rand[np.arange(0, round(num_ccl*0.9))]
    id_cell_test = id_rand[np.arange(round(num_ccl*0.9), num_ccl)]
    
    # prepare sample indices (selected CCLs x 1298 DepOIs)
    id_train = np.arange(0, 1298) + id_cell_train[0]*1298
    for y in id_cell_train:
        id_train = np.union1d(id_train, np.arange(0, 1298) + y*1298)
    id_test = np.arange(0, 1298) + id_cell_test[0] * 1298
    for y in id_cell_test:
        id_test = np.union1d(id_test, np.arange(0, 1298) + y*1298)
    print("\n\nTraining/validation on %d samples (%d CCLs x %d DepOIs) and testing on %d samples (%d CCLs x %d DepOIs).\n\n" % (
        len(id_train), len(id_cell_train), num_DepOI, len(id_test), len(id_cell_test), num_DepOI))

    # Full 4-omic DeepDEP model, composed of 6 sub-networks:
    # model_mut, model_exp, model_cna, model_meth: to learn data embedding of each omics
    # model_gene: to learn data embedding of gene fingerprints (involvement of a gene in 3115 functions)
    # model_final: to merge the above 5 sub-networks and predict gene-effect scores
    t = time.time()
    # subnetwork of mutations
    model_mut = models.Sequential()
    model_mut.add(Dense(output_dim=1000, input_dim=premodel_mut[0][0].shape[0], activation=activation_func,
                        weights=premodel_mut[0], trainable=True))
    model_mut.add(Dense(output_dim=100, input_dim=1000, activation=activation_func, weights=premodel_mut[1],
                        trainable=True))
    model_mut.add(Dense(output_dim=50, input_dim=100, activation=activation_func, weights=premodel_mut[2],
                        trainable=True))
    
    # subnetwork of expression
    model_exp = models.Sequential()
    model_exp.add(Dense(output_dim=500, input_dim=premodel_exp[0][0].shape[0], activation=activation_func,
                        weights=premodel_exp[0], trainable=True))
    model_exp.add(Dense(output_dim=200, input_dim=500, activation=activation_func, weights=premodel_exp[1],
                        trainable=True))
    model_exp.add(Dense(output_dim=50, input_dim=200, activation=activation_func, weights=premodel_exp[2],
                        trainable=True))
    
    # subnetwork of copy number alterations
    model_cna = models.Sequential()
    model_cna.add(Dense(output_dim=500, input_dim=premodel_cna[0][0].shape[0], activation=activation_func,
                        weights=premodel_cna[0], trainable=True))
    model_cna.add(Dense(output_dim=200, input_dim=500, activation=activation_func, weights=premodel_cna[1],
                        trainable=True))
    model_cna.add(Dense(output_dim=50, input_dim=200, activation=activation_func, weights=premodel_cna[2],
                        trainable=True))
    
    # subnetwork of DNA methylations
    model_meth = models.Sequential()
    model_meth.add(Dense(output_dim=500, input_dim=premodel_meth[0][0].shape[0], activation=activation_func,
                         weights=premodel_meth[0], trainable=True))
    model_meth.add(Dense(output_dim=200, input_dim=500, activation=activation_func, weights=premodel_meth[1],
                         trainable=True))
    model_meth.add(Dense(output_dim=50, input_dim=200, activation=activation_func, weights=premodel_meth[2],
                         trainable=True))
    
    # subnetwork of gene fingerprints
    model_gene = models.Sequential()
    model_gene.add(Dense(output_dim=1000, input_dim=data_fprint.shape[1], activation=activation_func, init=init,
                         trainable=True))
    model_gene.add(Dense(output_dim=100, input_dim=1000, activation=activation_func, init=init, trainable=True))
    model_gene.add(Dense(output_dim=50, input_dim=100, activation=activation_func, init=init, trainable=True))

    # prediction network
    model_final = models.Sequential()
    model_final.add(Merge([model_mut, model_exp, model_cna, model_meth, model_gene], mode='concat'))
    model_final.add(Dense(output_dim=dense_layer_dim, input_dim=250, activation=activation_func, init=init,
                          trainable=True))
    model_final.add(Dense(output_dim=dense_layer_dim, input_dim=dense_layer_dim, activation=activation_func, init=init,
                          trainable=True))
    model_final.add(Dense(output_dim=1, input_dim=dense_layer_dim, activation=activation_func2, init=init,
                          trainable=True))
    
    # training with early stopping with 3 patience
    history = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='min')
    model_final.compile(loss='mse', optimizer='adam')
    model_final.fit(
        [data_mut[id_train], data_exp[id_train], data_cna[id_train], data_meth[id_train], data_fprint[id_train]],
        data_dep[id_train], nb_epoch=num_epoch, validation_split=1/9, batch_size=batch_size, shuffle=True,
        callbacks=[history])
    cost_testing = model_final.evaluate(
        [data_mut[id_test], data_exp[id_test], data_cna[id_test], data_meth[id_test], data_fprint[id_test]],
        data_dep[id_test], verbose=0, batch_size=batch_size)
    print("\n\nFull DeepDEP model training completed in %.1f mins.\nloss:%.4f valloss:%.4f testloss:%.4f" % (
        (time.time() - t)/60, history.model.model.history.history['loss'][history.stopped_epoch],
        history.model.model.history.history['val_loss'][history.stopped_epoch], cost_testing))
    model_final.save("/results/models/model_demo.h5")
    print("\n\nFull DeepDEP model saved in /results/models/model_demo.h5\n\n")

    # mutation-alone model (Mut-DeepDEP)
    t = time.time()
    model_mut = models.Sequential()
    model_mut.add(Dense(output_dim=1000, input_dim=premodel_mut[0][0].shape[0], activation=activation_func,
                        weights=premodel_mut[0], trainable=True))
    model_mut.add(Dense(output_dim=100, input_dim=1000, activation=activation_func, weights=premodel_mut[1],
                        trainable=True))
    model_mut.add(Dense(output_dim=50, input_dim=100, activation=activation_func, weights=premodel_mut[2],
                        trainable=True))

    model_gene = models.Sequential()
    model_gene.add(Dense(output_dim=1000, input_dim=data_fprint.shape[1], activation=activation_func, init=init,
                         trainable=True))
    model_gene.add(Dense(output_dim=100, input_dim=1000, activation=activation_func, init=init, trainable=True))
    model_gene.add(Dense(output_dim=50, input_dim=100, activation=activation_func, init=init, trainable=True))

    model_final = models.Sequential()
    model_final.add(Merge([model_mut, model_gene], mode='concat'))
    model_final.add(Dense(output_dim=dense_layer_dim, input_dim=100, activation=activation_func, init=init))
    model_final.add(Dense(output_dim=dense_layer_dim, input_dim=dense_layer_dim, activation=activation_func, init=init))
    model_final.add(Dense(output_dim=1, input_dim=dense_layer_dim, activation=activation_func2, init=init))

    history = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='min')
    model_final.compile(loss='mse', optimizer='adam')
    model_final.fit([data_mut[id_train], data_fprint[id_train]], data_dep[id_train], nb_epoch=num_epoch,
                    validation_split=1/9, batch_size=batch_size, shuffle=True, callbacks=[history])
    cost_testing = model_final.evaluate([data_mut[id_test], data_fprint[id_test]], data_dep[id_test], verbose=0,
                                        batch_size=batch_size)
    print("\n\nMut-DeepDEP model training completed in %.1f mins.\nloss:%.4f valloss:%.4f testloss:%.4f" % (
        (time.time() - t)/60, history.model.model.history.history['loss'][history.stopped_epoch],
        history.model.model.history.history['val_loss'][history.stopped_epoch], cost_testing))
    model_final.save("/results/models/model_mut_demo.h5")
    print("\n\nMut-DeepDEP model saved in /results/models/model_mut_demo.h5\n\n")

    # expression-alone model (Exp-DeepDEP)
    t = time.time()
    model_exp = models.Sequential()
    model_exp.add(Dense(output_dim=500, input_dim=premodel_exp[0][0].shape[0], activation=activation_func,
                        weights=premodel_exp[0], trainable=True))
    model_exp.add(Dense(output_dim=200, input_dim=500, activation=activation_func, weights=premodel_exp[1],
                        trainable=True))
    model_exp.add(Dense(output_dim=50, input_dim=200, activation=activation_func, weights=premodel_exp[2],
                        trainable=True))

    model_gene = models.Sequential()
    model_gene.add(Dense(output_dim=1000, input_dim=data_fprint.shape[1], activation=activation_func, init=init,
                         trainable=True))
    model_gene.add(Dense(output_dim=100, input_dim=1000, activation=activation_func, init=init, trainable=True))
    model_gene.add(Dense(output_dim=50, input_dim=100, activation=activation_func, init=init, trainable=True))

    model_final = models.Sequential()
    model_final.add(Merge([model_exp, model_gene], mode='concat'))
    model_final.add(Dense(output_dim=dense_layer_dim, input_dim=100, activation=activation_func, init=init))
    model_final.add(Dense(output_dim=dense_layer_dim, input_dim=dense_layer_dim, activation=activation_func, init=init))
    model_final.add(Dense(output_dim=1, input_dim=dense_layer_dim, activation=activation_func2, init=init))

    history = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='min')
    model_final.compile(loss='mse', optimizer='adam')
    model_final.fit([data_exp[id_train], data_fprint[id_train]], data_dep[id_train], nb_epoch=num_epoch,
                    validation_split=1/9, batch_size=batch_size, shuffle=True, callbacks=[history])
    cost_testing = model_final.evaluate(
        [data_exp[id_test], data_fprint[id_test]], data_dep[id_test], verbose=0, batch_size=batch_size)
    print("\n\nExp-DeepDEP model training completed in %.1f mins.\nloss:%.4f valloss:%.4f testloss:%.4f" % (
        (time.time() - t)/60, history.model.model.history.history['loss'][history.stopped_epoch],
        history.model.model.history.history['val_loss'][history.stopped_epoch], cost_testing))
    model_final.save("/results/models/model_exp_demo.h5")
    print("\n\nExp-DeepDEP model saved in /results/models/model_exp_demo.h5\n\n")

    # methylation-alone model (Meth-DeepDEP)
    t = time.time()
    model_meth = models.Sequential()
    model_meth.add(Dense(output_dim=500, input_dim=premodel_meth[0][0].shape[0], activation=activation_func,
                         weights=premodel_meth[0], trainable=True))
    model_meth.add(Dense(output_dim=200, input_dim=500, activation=activation_func, weights=premodel_meth[1],
                         trainable=True))
    model_meth.add(Dense(output_dim=50, input_dim=200, activation=activation_func, weights=premodel_meth[2],
                         trainable=True))

    model_gene = models.Sequential()
    model_gene.add(Dense(output_dim=1000, input_dim=data_fprint.shape[1], activation=activation_func, init=init,
                         trainable=True))
    model_gene.add(Dense(output_dim=100, input_dim=1000, activation=activation_func, init=init, trainable=True))
    model_gene.add(Dense(output_dim=50, input_dim=100, activation=activation_func, init=init, trainable=True))

    model_final = models.Sequential()
    model_final.add(Merge([model_meth, model_gene], mode='concat'))
    model_final.add(Dense(output_dim=dense_layer_dim, input_dim=100, activation=activation_func, init=init))
    model_final.add(Dense(output_dim=dense_layer_dim, input_dim=dense_layer_dim, activation=activation_func, init=init))
    model_final.add(Dense(output_dim=1, input_dim=dense_layer_dim, activation=activation_func2, init=init))

    history = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='min')
    model_final.compile(loss='mse', optimizer='adam')
    model_final.fit([data_meth[id_train], data_fprint[id_train]], data_dep[id_train], nb_epoch=num_epoch,
                    validation_split=1/9, batch_size=batch_size, shuffle=True, callbacks=[history])
    cost_testing = model_final.evaluate([data_meth[id_test], data_fprint[id_test]], data_dep[id_test], verbose=0,
                                        batch_size=batch_size)
    print("\n\nMeth-DeepDEP model training completed in %.1f mins.\nloss:%.4f valloss:%.4f testloss:%.4f" % (
        (time.time() - t)/60, history.model.model.history.history['loss'][history.stopped_epoch],
        history.model.model.history.history['val_loss'][history.stopped_epoch], cost_testing))
    model_final.save("/results/models/model_meth_demo.h5")
    print("\n\nMeth-DeepDEP model saved in /results/models/model_meth_demo.h5\n\n")

    # copy number-alone model (CNA-DeepDEP)
    t = time.time()
    model_cna = models.Sequential()
    model_cna.add(Dense(output_dim=500, input_dim=premodel_cna[0][0].shape[0], activation=activation_func,
                        weights=premodel_cna[0], trainable=True))
    model_cna.add(Dense(output_dim=200, input_dim=500, activation=activation_func, weights=premodel_cna[1],
                        trainable=True))
    model_cna.add(Dense(output_dim=50, input_dim=200, activation=activation_func, weights=premodel_cna[2],
                        trainable=True))

    model_gene = models.Sequential()
    model_gene.add(Dense(output_dim=1000, input_dim=data_fprint.shape[1], activation=activation_func, init=init,
                         trainable=True))
    model_gene.add(Dense(output_dim=100, input_dim=1000, activation=activation_func, init=init, trainable=True))
    model_gene.add(Dense(output_dim=50, input_dim=100, activation=activation_func, init=init, trainable=True))

    model_final = models.Sequential()
    model_final.add(Merge([model_cna, model_gene], mode='concat'))
    model_final.add(Dense(output_dim=dense_layer_dim, input_dim=100, activation=activation_func, init=init))
    model_final.add(Dense(output_dim=dense_layer_dim, input_dim=dense_layer_dim, activation=activation_func, init=init))
    model_final.add(Dense(output_dim=1, input_dim=dense_layer_dim, activation=activation_func2, init=init))

    history = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='min')
    model_final.compile(loss='mse', optimizer='adam')
    model_final.fit([data_cna[id_train], data_fprint[id_train]], data_dep[id_train], nb_epoch=num_epoch,
                    validation_split=1/9, batch_size=batch_size, shuffle=True, callbacks=[history])
    cost_testing = model_final.evaluate([data_cna[id_test], data_fprint[id_test]], data_dep[id_test], verbose=0,
                                        batch_size=batch_size)
    print("\n\nCNA-DeepDEP model training completed in %.1f mins.\nloss:%.4f valloss:%.4f testloss:%.4f" % (
        (time.time() - t)/60, history.model.model.history.history['loss'][history.stopped_epoch],
        history.model.model.history.history['val_loss'][history.stopped_epoch], cost_testing))
    model_final.save("/results/models/model_cna_demo.h5")
    print("\n\nCNA-DeepDEP model saved in /results/models/model_cna_demo.h5\n\n")

    # 2-omics model (Mut/Exp-DeepDEP)
    t = time.time()
    model_mut = models.Sequential()
    model_mut.add(Dense(output_dim=1000, input_dim=premodel_mut[0][0].shape[0], activation=activation_func,
                        weights=premodel_mut[0], trainable=True))
    model_mut.add(Dense(output_dim=100, input_dim=1000, activation=activation_func, weights=premodel_mut[1],
                        trainable=True))
    model_mut.add(Dense(output_dim=50, input_dim=100, activation=activation_func, weights=premodel_mut[2],
                        trainable=True))

    model_exp = models.Sequential()
    model_exp.add(Dense(output_dim=500, input_dim=premodel_exp[0][0].shape[0], activation=activation_func,
                        weights=premodel_exp[0], trainable=True))
    model_exp.add(Dense(output_dim=200, input_dim=500, activation=activation_func, weights=premodel_exp[1],
                        trainable=True))
    model_exp.add(Dense(output_dim=50, input_dim=200, activation=activation_func, weights=premodel_exp[2],
                        trainable=True))

    model_gene = models.Sequential()
    model_gene.add(Dense(output_dim=1000, input_dim=data_fprint.shape[1], activation=activation_func, init=init,
                         trainable=True))
    model_gene.add(Dense(output_dim=100, input_dim=1000, activation=activation_func, init=init, trainable=True))
    model_gene.add(Dense(output_dim=50, input_dim=100, activation=activation_func, init=init, trainable=True))

    model_final = models.Sequential()
    model_final.add(Merge([model_mut, model_exp, model_gene], mode='concat'))
    model_final.add(Dense(output_dim=dense_layer_dim, input_dim=150, activation=activation_func, init=init))
    model_final.add(Dense(output_dim=dense_layer_dim, input_dim=dense_layer_dim, activation=activation_func, init=init))
    model_final.add(Dense(output_dim=1, input_dim=dense_layer_dim, activation=activation_func2, init=init))

    # 80% training, 10% validation, and 10% testing
    history = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='min')
    model_final.compile(loss='mse', optimizer='adam')
    model_final.fit([data_mut[id_train], data_exp[id_train], data_fprint[id_train]], data_dep[id_train],
                    nb_epoch=num_epoch, validation_split=1/9, batch_size=batch_size, shuffle=True, callbacks=[history])
    cost_testing = model_final.evaluate([data_mut[id_test], data_exp[id_test], data_fprint[id_test]], data_dep[id_test],
                                        verbose=0, batch_size=batch_size)
    print("\n\nMut/Exp-DeepDEP model training completed in %.1f mins.\nloss:%.4f valloss:%.4f testloss:%.4f" % (
        (time.time() - t)/60, history.model.model.history.history['loss'][history.stopped_epoch],
        history.model.model.history.history['val_loss'][history.stopped_epoch], cost_testing))
    model_final.save("/results/models/model_mutexp_demo.h5")
    print("\n\nMut/Exp-DeepDEP model saved in /results/models/model_mutexp_demo.h5\n\n")
