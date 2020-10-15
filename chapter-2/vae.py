from loguru import logger as log
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from scipy.stats import norm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras import backend as kb

class Sampling(layers.Layer):
    def __init__(self, stddev: float = 1.0, name = "sampling"):
        super(Sampling, self).__init__(name = name)
        self.__stddev = stddev

    def call(self, inputs) -> tf.Tensor:
        zmean, zlogvar = inputs
        batch = tf.shape(zmean)[0]
        dim = tf.shape(zmean)[1]
        eps: tf.Tensor = kb.random_normal(shape = (batch, dim), stddev = self.__stddev)
        return kb.exp(zlogvar/2) * eps + zmean

class Encoder(Model):
    def __init__(
        self,
        latentDim:         int,
        intermediateDim:   int,
        epsilonStd:        float = 1.0,
    ) -> None:
        super(Encoder, self).__init__()

        self.__intermediateLayer: layers.Layer = layers.Dense(
            intermediateDim,
            activation = 'relu',
            name = 'intermediate_layer',
        )

        self.__meanLayer: layers.Layer = layers.Dense(
            latentDim,
            name = 'mean_layer',
        )

        self.__logVarLayer: layers.Layer = layers.Dense(
            latentDim,
            name = 'log_var_layer',
        )

        self.__lambdaLayer: layers.Layer = Sampling(epsilonStd)

    def call(self, inputs) -> tf.Tensor:
        intermediateResult: tf.Tensor = self.__intermediateLayer(inputs)
        meanResult: tf.Tensor = self.__meanLayer(intermediateResult)
        logVarResult: tf.Tensor = self.__logVarLayer(intermediateResult)
        z = self.__lambdaLayer((meanResult, logVarResult))
        return meanResult, logVarResult, z

class Decoder(Model):
    def __init__(
        self,
        originalDim:       int,
        intermediateDim:   int,
    ) -> None:
        super(Decoder, self).__init__()

        self.__intermediateLayer: layers.Layer = layers.Dense(
            intermediateDim,
            activation = 'relu',
            name = 'decoder_intermediate',
            use_bias = True
        )

        self.__outputLayer: layers.Layer = layers.Dense(
            originalDim,
            activation = 'sigmoid',
            name = 'decoder_output'
        )
    
    def call(self, inputs) -> tf.Tensor:
        return self.__outputLayer(self.__intermediateLayer(inputs))

class VariationalAutoencoder(Model):
    def __init__(
        self,
        originalDim:       int,
        latentDim:         int,
        intermediateDim:   int,
        epsilonStd:        float = 1.0,
    ) -> None:
        super(VariationalAutoencoder, self).__init__()
        self.__originalDim = originalDim
        self.__encoder = Encoder(
            latentDim = 2,
            intermediateDim = 256,
            epsilonStd = epsilonStd
        )
        self.__decoder = Decoder(
            originalDim = 784,
            intermediateDim = 256
        )

    def call(self, inputs):
        zmean, zlogvar, z = self.__encoder(inputs)
        reconstructed = self.__decoder(z)
        klloss = -0.5 * tf.reduce_mean(
            zlogvar - tf.square(zmean) - tf.exp(zlogvar) + 1
        )
        self.add_loss(klloss)
        return reconstructed

    def encodeData(self, data: tf.Tensor, batchSize: int) -> tf.Tensor:
        return self.__encoder.predict(data, batch_size = batchSize)

    def decodeData(self, data: tf.Tensor) -> tf.Tensor:
        return self.__decoder.predict(data)

def prepareDataset() -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    (xtrain, _), (xtest, ytest) = tf.keras.datasets.mnist.load_data()
    xtrain = xtrain.reshape(60000, 784).astype("float32") / 255
    xtest = xtest.reshape(10000, 784).astype("float32") / 255
    return xtrain, xtest, ytest

def createVae(data) -> Model:
    vae = VariationalAutoencoder(
        originalDim = 784,
        latentDim = 2,
        intermediateDim = 256,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    vae.compile(
        optimizer = optimizer, 
        loss = tf.keras.losses.MeanSquaredError(),
    )

    vae.fit(
        data,
        data,
        batch_size = 64,
        epochs = 1
    )
    return vae

def encodeData(model: Model, datax: tf.Tensor, datay: tf.Tensor) -> None:
    _, _, z = model.encodeData(datax, batchSize = 64)
    plt.figure(figsize=(6, 6))
    plt.scatter(z[:,0], z[:,1], c=datay, cmap='viridis')
    plt.colorbar()
    plt.show()

def generate(model: Model) -> None:
    n = 15 
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = model.decodeData(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()

def main() -> None:
    log.info("hello!")
    xtrain, xtest, ytest = prepareDataset()
    # vae: tf.Model = createVae(xtrain)
    # encodeData(vae, xtest, ytest)
    # generate(vae)

    decoder = Decoder(784, 256)
    decoder.build(input_shape = (2, 1))
    decoder.summary()


if __name__ == "__main__":
    main()
