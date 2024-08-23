import numpy as np
import pandas as pd
import time
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from keras import models
import tensorflow as tf
import pickle

# Veri yükleme fonksiyonu
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

if __name__ == '__main__':
    gene = "BRCA1"

    # Model boyutlarını tanımla ve modeli yükle
    fprint_dim = 3115
    dense_layer_dim = 100

    # TCGA genomik verilerini ve gen fingerprint verilerini yükle
    data_mut_tcga, data_labels_mut_tcga, sample_names_mut_tcga, gene_names_mut_tcga = load_data("Data/CCL/ccl_mut_data_paired_with_tcga.txt")
    data_fprint_1298DepOIs, data_labels_fprint, gene_names_fprint, function_names_fprint = load_data("Data/crispr_gene_fingerprint_cgp.txt")
    print("\n\nDatasets successfully loaded.\n\n")

    premodel_mut = pickle.load(open(f'PytorchStaticSplits/OriginalCode/Results/Split2/USL_Pretrained/tcga_mut_ae_best_split_2.pickle', 'rb'))

    # Giriş katmanlarını tanımla
    input_mut = Input(shape=(premodel_mut[0][0].shape[0],))
    input_gene = Input(shape=(data_fprint_1298DepOIs.shape[1],))

    dims_mut = 4539
    activation_func = 'relu'
    activation_func2 = 'linear'
    init = 'he_uniform'

    # Keras modelini oluştur
    # Alt ağları tanımla
    model_mut = Dense(1000, activation=activation_func)(input_mut)
    model_mut = Dense(100, activation=activation_func)(model_mut)
    model_mut = Dense(50, activation=activation_func)(model_mut)

    model_gene = Dense(1000, activation=activation_func, kernel_initializer=init)(input_gene)
    model_gene = Dense(100, activation=activation_func, kernel_initializer=init)(model_gene)
    model_gene = Dense(50, activation=activation_func, kernel_initializer=init)(model_gene)

    # Alt ağları birleştir
    merged = Concatenate()([model_mut, model_gene])
    x = Dense(dense_layer_dim, activation=activation_func, kernel_initializer=init)(merged)
    x = Dense(dense_layer_dim, activation=activation_func, kernel_initializer=init)(x)
    output = Dense(1, activation=activation_func2, kernel_initializer=init)(x)

    # Modeli tanımla
    model = models.Model(inputs=[input_mut, input_gene], outputs=output)
    model.compile(optimizer='adam', loss='mse')

    # Model ağırlıklarını yükle
    model.load_weights("PytorchStaticSplits/OriginalCode/Results/Split2/PredictionNetworkModels/model_full_2_mutfingerprint.h5")

    # BRCA1 için analiz yap
    brca1_index = gene_names_mut_tcga.index(gene)
    data_pred = np.zeros((len(sample_names_mut_tcga), data_fprint_1298DepOIs.shape[0]))
    max_se_scores = []

    t = time.time()
    for z in np.arange(0, len(sample_names_mut_tcga)):
        # BRCA1 geninin mutasyon durumu (mutasyonlu ve mutasyonsuz)
        data_mut_batch = np.zeros((data_fprint_1298DepOIs.shape[0], dims_mut), dtype='float32')
        data_mut_batch[:, :] = data_mut_tcga[z, :]  # Diğer genlerin mutasyon durumunu koru
        
        # BRCA1 mutasyonlu durumu ve mutasyonsuz durumu için tahminler
        data_mut_batch[:, brca1_index] = 1.0
        output_mut = model.predict([data_mut_batch, data_fprint_1298DepOIs], batch_size=32)

        data_mut_batch[:, brca1_index] = 0.0
        output_wt = model.predict([data_mut_batch, data_fprint_1298DepOIs], batch_size=32)

        # SE skorları: mutasyonlu - mutasyonsuz
        se_scores = output_mut - output_wt
        data_pred[z] = np.transpose(se_scores)
        print("TCGA sample %d predicted..." % z)

        # Her TCGA örneği için en düşük SE skoruna sahip geni bul
        min_se_index = np.argmin(se_scores)
        max_se_scores.append((sample_names_mut_tcga[z], gene_names_fprint[min_se_index], se_scores[min_se_index][0]))

    # Sonuçları CSV dosyasına yaz
    data_pred_df = pd.DataFrame(data=np.transpose(data_pred), index=gene_names_fprint, columns=sample_names_mut_tcga)
    data_pred_df.to_csv(f"PytorchStaticSplits/SyntheticEssentiality/{gene}_only_mut_predictions_CCL_Original_DeepDep.csv", index_label='DepOI', float_format='%.4f')

    # Her TCGA örneği için en yüksek SE skoruna sahip gen ve SE skorunu kaydet (küçükten büyüğe sıralı)
    max_se_scores_sorted = sorted(max_se_scores, key=lambda x: x[2])
    max_se_scores_df = pd.DataFrame(max_se_scores_sorted, columns=['Sample', 'Gene', 'SE_Score'])
    max_se_scores_df.to_csv(f"PytorchStaticSplits/SyntheticEssentiality/Highest_SE_Scores_{gene}_CCL_DeepDep.csv", index=False)

    print(f"\n\nPrediction completed in %.1f mins.\nResults saved in PytorchStaticSplits/SyntheticEssentiality/{gene}_only_mut_predictions_CCL_Original_DeepDep.csv\n" % ((time.time()-t)/60))
    print(f"Highest SE scores saved in PytorchStaticSplits/SyntheticEssentiality/Highest_SE_Scores_{gene}_CCL_DeepDep.csv\n")
