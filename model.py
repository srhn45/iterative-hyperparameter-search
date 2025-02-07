import tensorflow as tf

 
class SuperbLayer(tf.keras.layers.Layer):
    """Custom layer with batch normalization, dropout, and activation."""
    
    def __init__(self, n_neurons, activation="swish", kernel_initializer="he_normal", dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.n_neurons = n_neurons
        self.activation_fn = activation
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.hidden = tf.keras.layers.Dense(self.n_neurons, activation=None, kernel_initializer=self.kernel_initializer)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.activations.get(self.activation_fn)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        super().build(input_shape)
    
    def call(self, inputs, training=True):
        Z = self.activation(self.batchnorm(self.hidden(inputs)))
        Z = self.dropout(Z, training=training)
        return Z

class SuperbModel(tf.keras.Model):
    def __init__(self, output_dim=10, n_layers=5, n_neurons=30, dropout_rate=0.2, activation="swish", **kwargs):
        super().__init__(**kwargs)
        
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.output_dim = output_dim
        self.hidden_layers = []
        self.out = None

    def build(self, input_shape):
        self.flatten = tf.keras.layers.Flatten()
        self.hidden_layers = [SuperbLayer(self.n_neurons, dropout_rate=self.dropout_rate, activation=self.activation) 
                              for _ in range(self.n_layers)]
        self.out = tf.keras.layers.Dense(self.output_dim, activation="softmax")
        super().build(input_shape)

    def call(self, inputs, training=True):
        Z = inputs
        Z = self.flatten(Z)
        for layer in self.hidden_layers:
            Z = layer(Z, training=training)
        return self.out(Z)