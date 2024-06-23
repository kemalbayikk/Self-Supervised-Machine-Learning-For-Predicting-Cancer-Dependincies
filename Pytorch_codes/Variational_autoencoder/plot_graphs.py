import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import seaborn as sns
import numpy as np
import pandas as pd

def plot_density(y_true_train, y_pred_train, y_pred_test, batch_size, learning_rate, epochs):
    plt.figure(figsize=(8, 6))
    sns.kdeplot(y_true_train, label='CCL original', color='blue')
    sns.kdeplot(y_pred_train, label='CCL predicted', color='orange')
    sns.kdeplot(y_pred_test, label='Tumor predicted', color='brown')
    plt.xlabel('Dependency score')
    plt.ylabel('Density (x0.01)')
    plt.title(f'Density plot of Dependency Scores\nBatch Size: {batch_size}, Learning Rate: {learning_rate}, Epochs: {epochs} VAE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/predictions/dependency_score_density_plot_{batch_size}_{learning_rate}_{epochs}_VAE_last.png')
    plt.show()

def plot_results(y_true_train, y_pred_train, y_true_test, y_pred_test, batch_size, learning_rate, epochs):
    plt.figure(figsize=(14, 6))

    # Training/validation plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred_train, y_true_train, alpha=0.5)
    coef_train = np.polyfit(y_pred_train, y_true_train, 1)
    poly1d_fn_train = np.poly1d(coef_train)
    plt.plot(np.unique(y_pred_train), poly1d_fn_train(np.unique(y_pred_train)), color='red')
    plt.xlabel('DeepDEP-predicted score')
    plt.ylabel('Original dependency score')
    plt.title(f'Training/validation\nBatch Size: {batch_size}, Learning Rate: {learning_rate}, Epochs: {epochs}, VAE')
    plt.xlim(-4, 5)
    plt.ylim(-4, 5)
    pearson_corr_train, _ = pearsonr(y_pred_train, y_true_train)
    mse_train = mean_squared_error(y_true_train, y_pred_train)
    plt.text(0.1, 0.9, f'$\\rho$ = {pearson_corr_train:.2f}\nMSE = {mse_train:.3f}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.8, f'y = {coef_train[0]:.2f}x + {coef_train[1]:.2f}', color='red', transform=plt.gca().transAxes)

    # Testing plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred_test, y_true_test, alpha=0.5)
    coef_test = np.polyfit(y_pred_test, y_true_test, 1)
    poly1d_fn_test = np.poly1d(coef_test)
    plt.plot(np.unique(y_pred_test), poly1d_fn_test(np.unique(y_pred_test)), color='red')
    plt.xlabel('DeepDEP-predicted score')
    plt.ylabel('Original dependency score')
    plt.title(f'Testing\nBatch Size: {batch_size}, Learning Rate: {learning_rate}, Epochs: {epochs}, VAE')
    plt.xlim(-4, 5)
    plt.ylim(-4, 5)
    pearson_corr_test, _ = pearsonr(y_pred_test, y_true_test)
    mse_test = mean_squared_error(y_true_test, y_pred_test)
    plt.text(0.1, 0.9, f'$\\rho$ = {pearson_corr_test:.2f}\nMSE = {mse_test:.3f}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.8, f'y = {coef_test[0]:.2f}x + {coef_test[1]:.2f}', color='red', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.savefig(f'results/predictions/prediction_scatter_plots_{batch_size}_{learning_rate}_{epochs}_VAE.png')
    plt.show()


y_true_train = np.loadtxt('results/predictions/y_true_train_CCL_VAE.txt', dtype=float)
y_pred_train = np.loadtxt('results/predictions/y_pred_train_CCL_VAE.txt', dtype=float)
y_true_test = np.loadtxt('results/predictions/y_true_test_CCL_VAE.txt', dtype=float)
y_pred_test = np.loadtxt('results/predictions/y_pred_test_CCL_VAE.txt', dtype=float)

# CSV dosyasının yolunu belirtin
csv_file_path = 'results/predictions/tcga_predicted_data_vae_model_demo.txt'

# CSV dosyasını pandas DataFrame olarak yükleyin
data_pred_df = pd.read_csv(csv_file_path, sep='\t', index_col='CRISPR_GENE')

# DataFrame'i numpy dizisine dönüştürün ve transpose işlemini geri alın
data_pred = data_pred_df.values.T

# data_pred dizisinin boyutlarını kontrol edin
print(data_pred.shape)

plot_density(y_true_train[0:len(y_true_train) - 1].flatten(),y_pred_train[0:len(y_pred_train) - 1].flatten(),data_pred.flatten(),5000,1e-4,20)
plot_results(y_true_train, y_pred_train, y_true_test, y_pred_test, 5000, 1e-4, 20)