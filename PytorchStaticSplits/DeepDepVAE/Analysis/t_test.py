from scipy.stats import ttest_rel, pearsonr
import numpy as np
import pandas as pd

# Dosya okuma fonksiyonu
def read_results_from_file(file_path):
    with open(file_path, 'r') as file:
        results = file.readlines()
    # Satır sonu karakterlerini temizleyip, float'a çeviriyoruz
    return [float(line.strip()) for line in results]

# Korelasyon sonuçlarını dosyaya yazma fonksiyonu
def write_correlations_to_file(correlations, file_path):
    with open(file_path, 'w') as file:
        for corr in correlations:
            file.write(f"{corr}\n")

# Model sonuçlarının dosya yollarını belirt
model1_file = 'PytorchStaticSplits/DeepDepVAE/Results/Split2/predictions/y_pred_test_Prediction_Network_VAE_Split_2_LR_0.001.txt'
model2_file = 'PytorchStaticSplits/OriginalCode/Results/Split2/y_pred_test_CCL_Split_2_Original.txt'
target_file = 'PytorchStaticSplits/DeepDepVAE/Results/Split2/predictions/y_true_test_CCL_VAE_Split_2.txt'  # Gerçek hedef değerlerinin bulunduğu dosya

# Sonuçları ve hedef değerleri dosyalardan oku
model1_results = read_results_from_file(model1_file)
model2_results = read_results_from_file(model2_file)
target_values = read_results_from_file(target_file)
# Her veri noktası için Pearson korelasyonu hesapla
model1_correlations = []
model2_correlations = []

# CSV dosya yollarını belirt
model1_csv_file = 'model1_differences.csv'
model2_csv_file = 'model2_differences.csv'
target_csv_file = 'target_values.csv'
differences_csv_file = 'model_differences.csv'

# Her veri noktası için farkları hesapla (model sonuçları - gerçek değerler)
model1_differences = [abs(model1_results[i] - target_values[i]) for i in range(len(model1_results))]
model2_differences = [abs(model2_results[i] - target_values[i]) for i in range(len(model2_results))]

# Farkları aynı CSV dosyasına kaydetme
def save_differences_to_csv(model1_diff, model2_diff, file_path):
    df = pd.DataFrame({
        'Model 1 Differences': model1_diff,
        'Model 2 Differences': model2_diff
    })
    df.to_csv(file_path, index=False)

# Farkları CSV dosyasına kaydet
save_differences_to_csv(model1_differences, model2_differences, differences_csv_file)

# Farklar üzerinde t testi uygulama
t_statistic, p_value = ttest_rel(model1_differences, model2_differences, alternative='greater')

# Sonuçları yazdırma
print(f"T-istatistiği: {t_statistic}")
print(f"P-değeri: {p_value}")

# P-değerinin istatistiksel olarak anlamlı olup olmadığını kontrol et
if p_value < 0.05:
    print("İki model arasındaki farklar istatistiksel olarak anlamlıdır.")
else:
    print("İki model arasındaki farklar istatistiksel olarak anlamlı değildir.")