import pickle
import pandas as pd
import numpy as np

# Pickle dosyasını yükleyin
with open('Data/ccl_complete_data_278CCL_1298DepOI_360844samples.pickle', 'rb') as f:
    data_mut, data_exp, data_cna, data_meth, data_dep, data_fprint = pickle.load(f)

datas = [data_mut, data_exp, data_cna, data_meth, data_dep, data_fprint]


for data in datas:
    # Veri türünü ve boyutlarını inceleyin
    print(f"Veri türü: {type(data)}")
    if isinstance(data, np.ndarray):
        print(f"Veri boyutları: {data.shape}")

        # Verilerin binary olup olmadığını kontrol etme
        is_binary = np.all((data == 0) | (data == 1))
        if is_binary:
            print("Veriler binary (0 veya 1).")
        else:
            print("Veriler binary değil.")

        # Verinin ilk birkaç satırını yazdırma
        print("Verinin ilk birkaç satırı:")
        print(data[:5])

    else:
        print("Veri numpy array değil.")

# data = data_exp

# # Veri türünü ve boyutlarını inceleyin
# print(f"Veri türü: {type(data)}")
# if isinstance(data, np.ndarray):
#     print(f"Veri boyutları: {data.shape}")

# # Numpy array'i DataFrame'e dönüştürme
# try:
#     if isinstance(data, np.ndarray):
#         # Eğer sütun adları varsa belirtin, yoksa varsayılan adları kullanın
#         column_names = [f"feature_{i}" for i in range(data.shape[1])]
#         data_df = pd.DataFrame(data, columns=column_names)
#     else:
#         raise ValueError("Beklenmeyen veri türü")

#     # DataFrame'i CSV dosyasına kaydedin
#     csv_file_path = 'Data/CSVs/data_mut_csv.csv'
#     data_df.to_csv(csv_file_path, index=False)
#     print(f"Veri başarıyla {csv_file_path} dosyasına kaydedildi.")
# except Exception as e:
#     print(f"Veri dönüştürme hatası: {e}")
