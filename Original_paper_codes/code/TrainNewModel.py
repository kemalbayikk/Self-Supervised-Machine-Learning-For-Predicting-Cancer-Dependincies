import pickle
from keras import models
from keras.layers import Dense, Concatenate, Input
from keras.callbacks import EarlyStopping
import numpy as np
import time

if __name__ == '__main__':
    # with open('Data/ccl_complete_data_28CCL_1298DepOI_36344samples_demo.pickle', 'rb') as f:
    #     data_mut, data_exp, data_cna, data_meth, data_dep, data_fprint = pickle.load(f)

    with open('Data/ccl_complete_data_278CCL_1298DepOI_360844samples.pickle', 'rb') as f:
        data_mut, data_exp, data_cna, data_meth, data_dep, data_fprint = pickle.load(f)

    premodel_mut = pickle.load(open('results/autoencoders/premodel_tcga_mut_1000_100_50.pickle', 'rb'))
    premodel_exp = pickle.load(open('results/autoencoders/premodel_tcga_exp_500_200_50.pickle', 'rb'))
    premodel_cna = pickle.load(open('results/autoencoders/premodel_tcga_cna_500_200_50.pickle', 'rb'))
    premodel_meth = pickle.load(open('results/autoencoders/premodel_tcga_meth_500_200_50.pickle', 'rb'))
    print("\n\nDatasets successfully loaded.")
    print(premodel_mut[0][0])
    print(premodel_mut[0][0].shape[0])

    activation_func = 'relu'
    activation_func2 = 'linear'
    kernel_initializer = 'he_uniform'
    dense_layer_dim = 250
    batch_size = 500
    num_epoch = 100
    num_DepOI = 1298
    num_ccl = int(data_mut.shape[0] / num_DepOI)
    
    id_rand = np.random.permutation(num_ccl)
    id_cell_train = id_rand[np.arange(0, round(num_ccl * 0.9))]
    id_cell_test = id_rand[np.arange(round(num_ccl * 0.9), num_ccl)]
    
    id_train = np.arange(0, 1298) + id_cell_train[0] * 1298
    for y in id_cell_train:
        id_train = np.union1d(id_train, np.arange(0, 1298) + y * 1298)
    id_test = np.arange(0, 1298) + id_cell_test[0] * 1298
    for y in id_cell_test:
        id_test = np.union1d(id_test, np.arange(0, 1298) + y * 1298)
    print("\n\nTraining/validation on %d samples (%d CCLs x %d DepOIs) and testing on %d samples (%d CCLs x %d DepOIs).\n\n" % (
        len(id_train), len(id_cell_train), num_DepOI, len(id_test), len(id_cell_test), num_DepOI))

    # Subnetwork of mutations
    input_mut = Input(shape=(premodel_mut[0][0].shape[0],))
    x_mut = Dense(units=1000, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(input_mut)
    x_mut = Dense(units=100, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(x_mut)
    x_mut = Dense(units=50, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(x_mut)
    
    # Subnetwork of expression
    input_exp = Input(shape=(premodel_exp[0][0].shape[0],))
    x_exp = Dense(units=500, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(input_exp)
    x_exp = Dense(units=200, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(x_exp)
    x_exp = Dense(units=50, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(x_exp)
    
    # Subnetwork of copy number alterations
    input_cna = Input(shape=(premodel_cna[0][0].shape[0],))
    x_cna = Dense(units=500, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(input_cna)
    x_cna = Dense(units=200, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(x_cna)
    x_cna = Dense(units=50, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(x_cna)
    
    # Subnetwork of DNA methylations
    input_meth = Input(shape=(premodel_meth[0][0].shape[0],))
    x_meth = Dense(units=500, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(input_meth)
    x_meth = Dense(units=200, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(x_meth)
    x_meth = Dense(units=50, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(x_meth)
    
    # Subnetwork of gene fingerprints
    input_gene = Input(shape=(data_fprint.shape[1],))
    x_gene = Dense(units=1000, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(input_gene)
    x_gene = Dense(units=100, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(x_gene)
    x_gene = Dense(units=50, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(x_gene)

    # Full 4-omic DeepDEP model
    merged = Concatenate()([x_mut, x_exp, x_cna, x_meth, x_gene])
    dense_1 = Dense(units=dense_layer_dim, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(merged)
    dense_2 = Dense(units=dense_layer_dim, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(dense_1)
    output = Dense(units=1, activation=activation_func2, kernel_initializer=kernel_initializer, trainable=True)(dense_2)

    model_final = models.Model(inputs=[input_mut, input_exp, input_cna, input_meth, input_gene], outputs=output)

    # from keras.utils import plot_model

    # Visualize the model
    # plot_model(model_final, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


    t = time.time()
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='min')
    model_final.compile(loss='mse', optimizer='adam')
    history = model_final.fit(
        [data_mut[id_train], data_exp[id_train], data_cna[id_train], data_meth[id_train], data_fprint[id_train]],
        data_dep[id_train], epochs=num_epoch, validation_split=1/9, batch_size=batch_size, shuffle=True,
        callbacks=[early_stopping])
    cost_testing = model_final.evaluate(
        [data_mut[id_test], data_exp[id_test], data_cna[id_test], data_meth[id_test], data_fprint[id_test]],
        data_dep[id_test], verbose=0, batch_size=batch_size)
    print("\n\nFull DeepDEP model training completed in %.1f mins.\nloss:%.4f valloss:%.4f testloss:%.4f" % (
        (time.time() - t)/60, history.history['loss'][early_stopping.stopped_epoch],
        history.history['val_loss'][early_stopping.stopped_epoch], cost_testing))
    model_final.save("./results/models/model_demo.h5")
    print("\n\nFull DeepDEP model saved in /results/models/model_demo.h5\n\n")

    #----------------- 

    # mutation-alone model (Mut-DeepDEP)
    # input_mut = Input(shape=(premodel_mut[0][0].shape[0],))
    # x_mut = Dense(units=1000, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(input_mut)
    # x_mut = Dense(units=100, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(x_mut)
    # x_mut = Dense(units=50, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(x_mut)
    
    # input_gene = Input(shape=(data_fprint.shape[1],))
    # x_gene = Dense(units=1000, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(input_gene)
    # x_gene = Dense(units=100, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(x_gene)
    # x_gene = Dense(units=50, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(x_gene)

    # merged = Concatenate()([x_mut, x_gene])
    # dense_1 = Dense(units=dense_layer_dim, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(merged)
    # dense_2 = Dense(units=dense_layer_dim, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(dense_1)
    # output = Dense(units=1, activation=activation_func2, kernel_initializer=kernel_initializer, trainable=True)(dense_2)

    # model_final = models.Model(inputs=[input_mut, input_gene], outputs=output)

    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='min')
    # model_final.compile(loss='mse', optimizer='adam')
    # history = model_final.fit([data_mut[id_train], data_fprint[id_train]], data_dep[id_train], epochs=num_epoch,
    #                 validation_split=1/9, batch_size=batch_size, shuffle=True, callbacks=[early_stopping])
    # cost_testing = model_final.evaluate([data_mut[id_test], data_fprint[id_test]], data_dep[id_test], verbose=0,
    #                                     batch_size=batch_size)
    # print("\n\nMut-DeepDEP model training completed in %.1f mins.\nloss:%.4f valloss:%.4f testloss:%.4f" % (
    #     (time.time() - t)/60, history.history['loss'][early_stopping.stopped_epoch],
    #     history.history['val_loss'][early_stopping.stopped_epoch], cost_testing))
    # model_final.save("./results/models/model_mut_demo.h5")
    # print("\n\nMut-DeepDEP model saved in ./results/models/model_mut_demo.h5\n\n")

    # # expression-alone model (Exp-DeepDEP)
    # t = time.time()
    # input_exp = Input(shape=(premodel_exp[0][0].shape[0],))
    # x_exp = Dense(units=500, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(input_exp)
    # x_exp = Dense(units=200, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(x_exp)
    # x_exp = Dense(units=50, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(x_exp)
    
    # input_gene = Input(shape=(data_fprint.shape[1],))
    # x_gene = Dense(units=1000, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(input_gene)
    # x_gene = Dense(units=100, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(x_gene)
    # x_gene = Dense(units=50, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(x_gene)

    # merged = Concatenate()([x_exp, x_gene])
    # dense_1 = Dense(units=dense_layer_dim, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(merged)
    # dense_2 = Dense(units=dense_layer_dim, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(dense_1)
    # output = Dense(units=1, activation=activation_func2, kernel_initializer=kernel_initializer, trainable=True)(dense_2)

    # model_final = models.Model(inputs=[input_exp, input_gene], outputs=output)

    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='min')
    # model_final.compile(loss='mse', optimizer='adam')
    # history = model_final.fit([data_exp[id_train], data_fprint[id_train]], data_dep[id_train], epochs=num_epoch,
    #                 validation_split=1/9, batch_size=batch_size, shuffle=True, callbacks=[early_stopping])
    # cost_testing = model_final.evaluate(
    #     [data_exp[id_test], data_fprint[id_test]], data_dep[id_test], verbose=0, batch_size=batch_size)
    # print("\n\nExp-DeepDEP model training completed in %.1f mins.\nloss:%.4f valloss:%.4f testloss:%.4f" % (
    #     (time.time() - t)/60, history.history['loss'][early_stopping.stopped_epoch],
    #     history.history['val_loss'][early_stopping.stopped_epoch], cost_testing))
    # model_final.save("./results/models/model_exp_demo.h5")
    # print("\n\nExp-DeepDEP model saved in ./results/models/model_exp_demo.h5\n\n")

    # # methylation-alone model (Meth-DeepDEP)
    # t = time.time()
    # input_meth = Input(shape=(premodel_meth[0][0].shape[0],))
    # x_meth = Dense(units=500, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(input_meth)
    # x_meth = Dense(units=200, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(x_meth)
    # x_meth = Dense(units=50, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(x_meth)
    
    # input_gene = Input(shape=(data_fprint.shape[1],))
    # x_gene = Dense(units=1000, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(input_gene)
    # x_gene = Dense(units=100, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(x_gene)
    # x_gene = Dense(units=50, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(x_gene)

    # merged = Concatenate()([x_meth, x_gene])
    # dense_1 = Dense(units=dense_layer_dim, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(merged)
    # dense_2 = Dense(units=dense_layer_dim, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(dense_1)
    # output = Dense(units=1, activation=activation_func2, kernel_initializer=kernel_initializer, trainable=True)(dense_2)

    # model_final = models.Model(inputs=[input_meth, input_gene], outputs=output)

    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='min')
    # model_final.compile(loss='mse', optimizer='adam')
    # history = model_final.fit([data_meth[id_train], data_fprint[id_train]], data_dep[id_train], epochs=num_epoch,
    #                 validation_split=1/9, batch_size=batch_size, shuffle=True, callbacks=[early_stopping])
    # cost_testing = model_final.evaluate([data_meth[id_test], data_fprint[id_test]], data_dep[id_test], verbose=0,
    #                                     batch_size=batch_size)
    # print("\n\nMeth-DeepDEP model training completed in %.1f mins.\nloss:%.4f valloss:%.4f testloss:%.4f" % (
    #     (time.time() - t)/60, history.history['loss'][early_stopping.stopped_epoch],
    #     history.history['val_loss'][early_stopping.stopped_epoch], cost_testing))
    # model_final.save("./results/models/model_meth_demo.h5")
    # print("\n\nMeth-DeepDEP model saved in ./results/models/model_meth_demo.h5\n\n")

    # # copy number-alone model (CNA-DeepDEP)
    # t = time.time()
    # input_cna = Input(shape=(premodel_cna[0][0].shape[0],))
    # x_cna = Dense(units=500, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(input_cna)
    # x_cna = Dense(units=200, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(x_cna)
    # x_cna = Dense(units=50, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(x_cna)
    
    # input_gene = Input(shape=(data_fprint.shape[1],))
    # x_gene = Dense(units=1000, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(input_gene)
    # x_gene = Dense(units=100, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(x_gene)
    # x_gene = Dense(units=50, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(x_gene)

    # merged = Concatenate()([x_cna, x_gene])
    # dense_1 = Dense(units=dense_layer_dim, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(merged)
    # dense_2 = Dense(units=dense_layer_dim, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(dense_1)
    # output = Dense(units=1, activation=activation_func2, kernel_initializer=kernel_initializer, trainable=True)(dense_2)

    # model_final = models.Model(inputs=[input_cna, input_gene], outputs=output)

    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='min')
    # model_final.compile(loss='mse', optimizer='adam')
    # history = model_final.fit([data_cna[id_train], data_fprint[id_train]], data_dep[id_train], epochs=num_epoch,
    #                 validation_split=1/9, batch_size=batch_size, shuffle=True, callbacks=[early_stopping])
    # cost_testing = model_final.evaluate([data_cna[id_test], data_fprint[id_test]], data_dep[id_test], verbose=0,
    #                                     batch_size=batch_size)
    # print("\n\nCNA-DeepDEP model training completed in %.1f mins.\nloss:%.4f valloss:%.4f testloss:%.4f" % (
    #     (time.time() - t)/60, history.history['loss'][early_stopping.stopped_epoch],
    #     history.history['val_loss'][early_stopping.stopped_epoch], cost_testing))
    # model_final.save("./results/models/model_cna_demo.h5")
    # print("\n\nCNA-DeepDEP model saved in ./results/models/model_cna_demo.h5\n\n")

    # # 2-omics model (Mut/Exp-DeepDEP)
    # t = time.time()
    # input_mut = Input(shape=(premodel_mut[0][0].shape[0],))
    # x_mut = Dense(units=1000, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(input_mut)
    # x_mut = Dense(units=100, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(x_mut)
    # x_mut = Dense(units=50, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(x_mut)
    
    # input_exp = Input(shape=(premodel_exp[0][0].shape[0],))
    # x_exp = Dense(units=500, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(input_exp)
    # x_exp = Dense(units=200, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(x_exp)
    # x_exp = Dense(units=50, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(x_exp)
    
    # input_gene = Input(shape=(data_fprint.shape[1],))
    # x_gene = Dense(units=1000, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(input_gene)
    # x_gene = Dense(units=100, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(x_gene)
    # x_gene = Dense(units=50, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(x_gene)

    # merged = Concatenate()([x_mut, x_exp, x_gene])
    # dense_1 = Dense(units=dense_layer_dim, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(merged)
    # dense_2 = Dense(units=dense_layer_dim, activation=activation_func, kernel_initializer=kernel_initializer, trainable=True)(dense_1)
    # output = Dense(units=1, activation=activation_func2, kernel_initializer=kernel_initializer, trainable=True)(dense_2)

    # model_final = models.Model(inputs=[input_mut, input_exp, input_gene], outputs=output)

    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='min')
    # model_final.compile(loss='mse', optimizer='adam')
    # history = model_final.fit([data_mut[id_train], data_exp[id_train], data_fprint[id_train]], data_dep[id_train],
    #                 epochs=num_epoch, validation_split=1/9, batch_size=batch_size, shuffle=True, callbacks=[early_stopping])
    # cost_testing = model_final.evaluate([data_mut[id_test], data_exp[id_test], data_fprint[id_test]], data_dep[id_test],
    #                                     verbose=0, batch_size=batch_size)
    # print("\n\nMut/Exp-DeepDEP model training completed in %.1f mins.\nloss:%.4f valloss:%.4f testloss:%.4f" % (
    #     (time.time() - t)/60, history.history['loss'][early_stopping.stopped_epoch],
    #     history.history['val_loss'][early_stopping.stopped_epoch], cost_testing))
    # model_final.save("./results/models/model_mutexp_demo.h5")
    # print("\n\nMut/Exp-DeepDEP model saved in ./results/models/model_mutexp_demo.h5\n\n")
