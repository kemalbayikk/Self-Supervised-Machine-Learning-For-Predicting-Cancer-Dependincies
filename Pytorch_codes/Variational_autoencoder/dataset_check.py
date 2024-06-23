import pickle
import numpy as np
with open('Data/ccl_complete_data_278CCL_1298DepOI_360844samples.pickle', 'rb') as f:
        data_mut, data_exp, data_cna, data_meth, data_dep, data_fprint = pickle.load(f)

def is_binary_array_np(arr):
    arr = np.array(arr)  # Eğer zaten NumPy dizisi değilse, dönüştür
    # Dizideki tüm değerlerin 0 veya 1 olduğunu kontrol et
    return np.all((arr == 0) | (arr == 1))

result_mut = is_binary_array_np(data_mut)
print(f"Mut: {result_mut}")
result_exp = is_binary_array_np(data_exp)
print(f"Exp: {result_exp}")
result_cna = is_binary_array_np(data_cna)
print(f"Cna: {result_cna}")
result_meth = is_binary_array_np(data_meth)
print(f"Meth: {result_meth}")
result_fprint = is_binary_array_np(data_fprint)
print(f"Fprint: {result_fprint}")