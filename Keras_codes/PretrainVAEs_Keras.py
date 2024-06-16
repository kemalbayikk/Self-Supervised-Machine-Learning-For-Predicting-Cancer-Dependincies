import pickle
from keras import models, layers
from keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np
import time
import os

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

    print(data)

    return data, data_labels, sample_names, gene_names

def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAELossLayer(layers.Layer):
    def __init__(self, z_log_var, z_mean, **kwargs):
        self.is_placeholder = True
        self.z_log_var = z_log_var
        self.z_mean = z_mean
        super(VAELossLayer, self).__init__(**kwargs)

    def vae_loss(self, inputs, outputs):
        reconstruction_loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(inputs, outputs), axis=-1)
        kl_loss = 1 + self.z_log_var - tf.square(self.z_mean) - tf.exp(self.z_log_var)
        kl_loss = tf.reduce_sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return tf.reduce_mean(reconstruction_loss + kl_loss)

    def call(self, inputs):
        outputs = inputs[0]
        inputs = inputs[1]
        loss = self.vae_loss(inputs, outputs)
        self.add_loss(loss, inputs=inputs)
        return inputs

def VAE_dense_3layers(input_dim, first_layer_dim, second_layer_dim, latent_dim, activation_func, init='he_uniform'):
    print('input_dim = ', input_dim)
    print('first_layer_dim = ', first_layer_dim)
    print('second_layer_dim = ', second_layer_dim)
    print('latent_dim = ', latent_dim)
    print('init = ', init)
    
    # Encoder
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(first_layer_dim, activation=activation_func, kernel_initializer=init)(inputs)
    x = layers.Dense(second_layer_dim, activation=activation_func, kernel_initializer=init)(x)
    z_mean = layers.Dense(latent_dim, activation='linear')(x)
    z_log_var = layers.Dense(latent_dim, activation='linear')(x)

    z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    
    encoder = models.Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    
    # Decoder
    latent_inputs = layers.Input(shape=(latent_dim,), name='z_sampling')
    x = layers.Dense(second_layer_dim, activation=activation_func, kernel_initializer=init)(latent_inputs)
    x = layers.Dense(first_layer_dim, activation=activation_func, kernel_initializer=init)(x)
    outputs = layers.Dense(input_dim, activation='sigmoid')(x)
    
    decoder = models.Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    
    # VAE Model
    outputs = decoder(encoder(inputs)[2])
    vae_loss_layer = VAELossLayer(z_log_var=z_log_var, z_mean=z_mean)([outputs, inputs])
    vae = models.Model(inputs, vae_loss_layer, name='vae_mlp')
    
    vae.compile(optimizer='adam')
    
    return vae

def save_weight_to_pickle(model, file_name):
    print('saving weights')

    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
        os.chmod(directory, mode=0o755)
        print(f"Directory {directory} created.")

    weight_list = []
    for layer in model.layers:
        weight_list.append(layer.get_weights())
    with open(file_name, 'wb') as handle:
        pickle.dump(weight_list, handle)
        
if __name__ == '__main__':
    filepath = "Data/TCGA/tcga_cna_data_paired_with_ccl.txt"
    # load TCGA mutation data, substitute here with other genomics
    data_mut_tcga, data_labels_mut_tcga, sample_names_mut_tcga, gene_names_mut_tcga = load_data(filepath)
    print("\n\nDatasets successfully loaded.")
    
    samples_to_predict = np.arange(0, 50)  # Sadece 50 örnek ile demo
    # predict the first 50 samples for DEMO ONLY, for all samples please substitute 50 by data_mut_tcga.shape[0]
    # prediction results of all 8238 TCGA samples can be found in /data/premodel_tcga_*.pickle
    
    input_dim = data_mut_tcga.shape[1]
    first_layer_dim = 1000
    second_layer_dim = 100
    latent_dim = 50
    batch_size = 64
    epoch_size = 100
    activation_function = 'relu'
    init = 'he_uniform'
    model_save_name = "premodel_tcga_mut_vae_%d_%d_%d" % (first_layer_dim, second_layer_dim, latent_dim)

    t = time.time()
    model = VAE_dense_3layers(input_dim=input_dim, first_layer_dim=first_layer_dim, second_layer_dim=second_layer_dim, latent_dim=latent_dim, activation_func=activation_function, init=init)
    model.fit(data_mut_tcga[samples_to_predict], data_mut_tcga[samples_to_predict], epochs=epoch_size, batch_size=batch_size, shuffle=True)
    
    cost = model.evaluate(data_mut_tcga[samples_to_predict], data_mut_tcga[samples_to_predict], verbose=0)
    print('\n\nVAE training completed in %.1f mins.\n with test loss:%.4f' % ((time.time()-t)/60, cost))
    
    save_weight_to_pickle(model, './results/autoencoders/' + model_save_name + '.pickle')
    print("\nResults saved in /results/autoencoders/%s.pickle\n\n" % model_save_name)
