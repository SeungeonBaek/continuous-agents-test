import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from tensorflow.keras import initializers, regularizers
from tensorflow.keras.layers import Layer

class gSDENoiseModel(Layer):
    """
    noise model for gSDE agent
    """
    def __init__(self, units):
        super(gSDENoiseModel,self).__init__()
        self.units = units

        self.kernel_initializer = initializers.he_normal()
        self.kernel_regularizer = regularizers.l2(l=0.005)

    def build(self, input_shape):
        super(gSDENoiseModel, self).build(input_shape)

        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True)

        self.b = self.add_weight(
            shape=(self.units, ),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True)

        self.epsilon = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=False)
        
        self.sample_weights()

    def call(self, latent):
        return tf.math.add(tf.matmul(latent, self.epsilon), self.b)

    def get_std(self):
        return tf.where(
            self.kernel <= 0,
            tf.exp(self.kernel),
            tf.math.log1p(self.kernel + 1e-6) + 1.0,
        )
    
    def sample_weights(self):
        w_dist = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros_like(self.kernel), scale_diag = self.get_std()
        )
        self.epsilon.assign(w_dist.sample())


class IDACGaussActorNoiseModel(Layer):
    """
    noise model for GaussianActor in IDAC agent
    """
    def __init__(self, units):
        super(IDACGaussActorNoiseModel,self).__init__()
        self.units = units

        self.kernel_initializer = initializers.he_normal()
        self.kernel_regularizer = regularizers.l2(l=0.005)

    def build(self, input_shape):
        super(IDACGaussActorNoiseModel, self).build(input_shape)

        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True)

        self.b = self.add_weight(
            shape=(self.units, ),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True)

        self.epsilon = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=False)
        
        self.sample_weights()

    def call(self, state):
        return tf.math.add(tf.matmul(state, self.epsilon), self.b)

    def get_std(self):
        return tf.where(
            self.kernel <= 0,
            tf.exp(self.kernel),
            tf.math.log1p(self.kernel + 1e-6) + 1.0,
        )
    
    def sample_weights(self):
        w_dist = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros_like(self.kernel), scale_diag = self.get_std()
        )
        self.epsilon.assign(w_dist.sample())


class IDACImplicitIActorNoiseModel(Layer):
    """
    noise model for ImplicitActor in IDAC agent
    """
    def __init__(self, units):
        super(IDACImplicitIActorNoiseModel,self).__init__()
        self.units = units

        self.kernel_initializer = initializers.he_normal()
        self.kernel_regularizer = regularizers.l2(l=0.005)

    def build(self, input_shape):
        super(IDACImplicitIActorNoiseModel, self).build(input_shape)

        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True)

        self.b = self.add_weight(
            shape=(self.units, ),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True)

        self.epsilon = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=False)
        
        self.sample_weights()

    def call(self, latent):
        return tf.math.add(tf.matmul(latent, self.epsilon), self.b)

    def get_std(self):
        return tf.where(
            self.kernel <= 0,
            tf.exp(self.kernel),
            tf.math.log1p(self.kernel + 1e-6) + 1.0,
        )
    
    def sample_weights(self):
        w_dist = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros_like(self.kernel), scale_diag = self.get_std()
        )
        self.epsilon.assign(w_dist.sample())


class IDACDistCriticNoiseModel(Layer):
    """
    noise model for Distributional Critic in IDAC agent
    """
    def __init__(self, units):
        super(IDACDistCriticNoiseModel,self).__init__()
        self.units = units

        self.kernel_initializer = initializers.he_normal()
        self.kernel_regularizer = regularizers.l2(l=0.005)

    def build(self, input_shape):
        super(IDACDistCriticNoiseModel, self).build(input_shape)

        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True)

        self.b = self.add_weight(
            shape=(self.units, ),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True)

        self.epsilon = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=False)
        
        self.sample_weights()

    def call(self, latent):
        return tf.math.add(tf.matmul(latent, self.epsilon), self.b)

    def get_std(self):
        return tf.where(
            self.kernel <= 0,
            tf.exp(self.kernel),
            tf.math.log1p(self.kernel + 1e-6) + 1.0,
        )
    
    def sample_weights(self):
        w_dist = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros_like(self.kernel), scale_diag = self.get_std()
        )
        self.epsilon.assign(w_dist.sample())


class gSDENoiseModel(Layer):
    """
    noise model for gSDE agent
    """
    def __init__(self, units):
        super(gSDENoiseModel,self).__init__()
        self.units = units

        self.kernel_initializer = initializers.he_normal()
        self.kernel_regularizer = regularizers.l2(l=0.005)

    def build(self, input_shape):
        super(gSDENoiseModel, self).build(input_shape)

        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True)

        self.b = self.add_weight(
            shape=(self.units, ),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True)

        self.epsilon = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=False)
        
        self.sample_weights()

    def call(self, latent):
        return tf.math.add(tf.matmul(latent, self.epsilon), self.b)

    def get_std(self):
        return tf.where(
            self.kernel <= 0,
            tf.exp(self.kernel),
            tf.math.log1p(self.kernel + 1e-6) + 1.0,
        )
    
    def sample_weights(self):
        w_dist = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros_like(self.kernel), scale_diag = self.get_std()
        )
        self.epsilon.assign(w_dist.sample())