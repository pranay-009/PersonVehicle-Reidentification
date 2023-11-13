from keras import backend as K
from keras.losses import Loss
import tensorflow as tf

"""
Custom Loss Fuction to handle Imbalanced weights

"""
class WeightedCategoricalCrossentropy(Loss):
    def __init__(self, class_weights):
        super(WeightedCategoricalCrossentropy, self).__init__()
        self.class_weights = class_weights

    def call(self, y_true, y_pred):
        # Ensure the predicted values are within a small range to avoid log(0) issues
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

        # Calculate the weighted cross-entropy loss
        weighted_losses = -tf.reduce_sum(y_true * tf.math.log(y_pred) * self.class_weights, axis=-1)

        return tf.reduce_mean(weighted_losses)

    
