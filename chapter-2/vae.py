from loguru import logger as log
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

class Encoder(layers.Layer):
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

class Decoder(layers.Layer):
    def __init__(
        self,
        originalDim:       int,
        intermediateDim:   int,
    ) -> None:
        super(Decoder, self).__init__()

        self.__intermediateLayer: layers.Layer = layers.Dense(
            intermediateDim,
            activation = 'relu',
            name = 'decoder_intermediate'
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

def prepareDataset() -> tf.Tensor:
    (xtrain, _), _ = tf.keras.datasets.mnist.load_data()
    xtrain = xtrain.reshape(60000, 784).astype("float32") / 255
    return xtrain

def main() -> None:
    log.info("hello!")

    vae = VariationalAutoencoder(
        originalDim = 784,
        latentDim = 2,
        intermediateDim = 256,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    vae.compile(
        optimizer = optimizer, 
        loss = tf.keras.losses.MeanSquaredError(),
        # metrics = [tf.keras.metrics.Mean()]
    )
    xtrain = prepareDataset()
    vae.fit(
        xtrain,
        xtrain,
        batch_size = 64,
        epochs = 2
    )

if __name__ == "__main__":
    main()
