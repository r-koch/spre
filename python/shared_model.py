import tensorflow as tf
from keras.saving import register_keras_serializable
from tensorflow.keras import layers  # type: ignore


@register_keras_serializable(package="spre")
class MeanOverSymbols(layers.Layer):
    # Reduces (batch, time, symbols, embed) -> (batch, time, embed)
    # by averaging over the symbol axis.

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=2)

    def compute_output_shape(self, input_shape):
        # input_shape: (batch, time, symbols, embed)
        return (input_shape[0], input_shape[1], input_shape[3])
