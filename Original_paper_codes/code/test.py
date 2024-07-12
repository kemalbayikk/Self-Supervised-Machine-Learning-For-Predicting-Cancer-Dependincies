import numpy as np
from keras.models import load_model
from scipy.stats import pearsonr
import pickle

if __name__ == '__main__':
    # Test veri kümesini yükleme
    with open('Data/ccl_complete_data_278CCL_1298DepOI_360844samples.pickle', 'rb') as f:
        data_mut, data_exp, data_cna, data_meth, data_dep, data_fprint = pickle.load(f)

    # Modeli yükleme
    model = load_model('./results/original_paper/model_demo.h5', custom_objects={'mse': 'mean_squared_error'})

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
    
    print("\n\nTesting on %d samples (%d CCLs x %d DepOIs).\n\n" % (
        len(id_test), len(id_cell_test), num_DepOI))
    
    # Tahminler ve gerçek değerler
    predictions = model.predict([data_mut[id_test], data_exp[id_test], data_cna[id_test], data_meth[id_test], data_fprint[id_test]])
    true_values = data_dep[id_test]

    # Pearson korelasyonu hesaplama
    predictions = predictions.flatten()
    true_values = true_values.flatten()
    pearson_corr, _ = pearsonr(predictions, true_values)

    print(f"Pearson Correlation: {pearson_corr}")