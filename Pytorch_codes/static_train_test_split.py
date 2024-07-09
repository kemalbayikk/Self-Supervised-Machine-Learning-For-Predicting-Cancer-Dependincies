import pickle
from sklearn.model_selection import train_test_split
import os

# Dosya adını belirtin
file_name = 'Data/ccl_complete_data_278CCL_1298DepOI_360844samples.pickle'

# Pickle dosyasını yükleyin
with open(file_name, 'rb') as f:
    data = pickle.load(f)

# Eğitim ve test setlerini saklayacağınız dizini oluşturun
output_dir = 'Data/train_test_splits'
os.makedirs(output_dir, exist_ok=True)

# 5 farklı train-test seti oluşturun ve kaydedin
train_data, test_data = train_test_split(data, test_size=0.1, random_state=1)

train_file = os.path.join(output_dir, 'train_data_split_1.pickle')
test_file = os.path.join(output_dir, 'test_data_split_1.pickle')

with open(train_file, 'wb') as f:
    pickle.dump(train_data, f)
with open(test_file, 'wb') as f:
    pickle.dump(test_data, f)

print(f'Train set 1 saved to {train_file}')
print(f'Test set 1 saved to {test_file}')