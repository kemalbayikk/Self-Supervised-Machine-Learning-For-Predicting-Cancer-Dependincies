import pickle
import pandas as pd
import numpy as np

# Pickle dosyasını açma ve verileri yükleme
with open('Data/ccl_complete_data_278CCL_1298DepOI_360844samples.pickle', 'rb') as f:
        data_mut, data_exp, data_cna, data_meth, data_dep, data_fprint = pickle.load(f)

# Numpy dizilerini pandas DataFrame'lere dönüştürme ve CSV dosyalarına kaydetme
# if isinstance(train_dataset, np.ndarray):
#     pd.DataFrame(train_dataset).to_csv('Data/data_mut.csv', index=False)
# if isinstance(data_exp, np.ndarray):
#     pd.DataFrame(data_exp).to_csv('Data/data_exp.csv', index=False)
# if isinstance(data_cna, np.ndarray):
#     pd.DataFrame(data_cna).to_csv('Data/data_cna.csv', index=False)
# if isinstance(data_meth, np.ndarray):
#     pd.DataFrame(data_meth).to_csv('Data/data_meth.csv', index=False)
if isinstance(data_dep, np.ndarray):
    pd.DataFrame(data_dep).to_csv('Data/data_dep.csv', index=False)
# if isinstance(data_fprint, np.ndarray):
#     pd.DataFrame(data_fprint).to_csv('Data/data_fprint.csv', index=False)

print("Veriler CSV dosyalarına başarıyla kaydedildi.")
