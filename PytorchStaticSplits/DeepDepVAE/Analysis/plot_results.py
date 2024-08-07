import numpy as np
import matplotlib.pyplot as plt

# Verileri yükleme
y_true_train = np.loadtxt('PytorchStaticSplits/DeepDepVAE/Results/Split2/predictions/y_true_train_Prediction_Network_VAE_Split_2_LR_0.001.txt', dtype=float)
y_pred_train = np.loadtxt('PytorchStaticSplits/DeepDepVAE/Results/Split2/predictions/y_pred_train_Prediction_Network_VAE_Split_2_LR_0.001.txt', dtype=float)
y_true_test = np.loadtxt('PytorchStaticSplits/DeepDepVAE/Results/Split2/predictions/y_true_test_Prediction_Network_VAE_Split_2_LR_0.001.txt', dtype=float)
y_pred_test = np.loadtxt('PytorchStaticSplits/DeepDepVAE/Results/Split2/predictions/y_pred_test_Prediction_Network_VAE_Split_2_LR_0.001.txt', dtype=float)

# Verileri yükleme
y_true_train_original = np.loadtxt('PytorchStaticSplits/OriginalCode/Results/Split2/y_true_train_CCL_Split_2_Original.txt', dtype=float)
y_pred_train_original = np.loadtxt('PytorchStaticSplits/OriginalCode/Results/Split2/y_pred_train_CCL_Split_2_Original.txt', dtype=float)
y_true_test_original = np.loadtxt('PytorchStaticSplits/OriginalCode/Results/Split2/y_true_test_CCL_Split_2_Original.txt', dtype=float)
y_pred_test_original = np.loadtxt('PytorchStaticSplits/OriginalCode/Results/Split2/y_pred_test_CCL_Split_2_Original.txt', dtype=float)

# Uzaklıkları hesaplama
distance_train = np.abs(y_true_train - y_pred_train) / np.sqrt(2)
distance_test = np.abs(y_true_test - y_pred_test) / np.sqrt(2)

# Grafiği oluşturma
plt.figure(figsize=(10, 8))

# # Eğitim verileri
# plt.scatter(y_pred_train, y_true_train, alpha=0.5, label='Train Data', color='blue')
# for i in range(len(y_true_train)):
#     plt.plot([y_pred_train[i], y_true_train[i]], [y_true_train[i], y_true_train[i]], 'b-', alpha=0.5)

# Orjinal Test verileri
plt.scatter(y_pred_test_original, y_true_test_original, alpha=0.5, label='Original Model', color='blue')
for i in range(len(y_true_test)):
    plt.plot([y_pred_test_original[i], y_true_test_original[i]], [y_true_test_original[i], y_true_test_original[i]], 'b-', alpha=0.5)

# Test verileri
plt.scatter(y_pred_test, y_true_test, alpha=0.5, label='VAE DeepDep', color='green')
for i in range(len(y_true_test)):
    plt.plot([y_pred_test[i], y_true_test[i]], [y_true_test[i], y_true_test[i]], 'g-', alpha=0.5)

# y = x doğrusunu ekleme
plt.plot(np.linspace(-4, 5, 100), np.linspace(-4, 5, 100), 'r--', label='y = x')

plt.title('VAE-DeepDep and Original-DeepDep Model Comparisons')
plt.xlabel('Predicted dependency score')
plt.ylabel('Original dependency score')
plt.legend()
plt.grid(True)

plt.show()
