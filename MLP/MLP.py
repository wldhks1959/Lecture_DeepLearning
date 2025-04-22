# MLP.py
import tensorflow as tf

class MLP:
    def __init__(self, hidden_layer_conf, num_output_nodes):
        self.hidden_layer_conf = hidden_layer_conf
        self.num_output_nodes = num_output_nodes
        self.logic_op_model = None

    def build_model(self):

        input_layer = tf.keras.Input(shape=[2, ])
        hidden_layers = input_layer

        if self.hidden_layer_conf is not None:
            for num_hidden_nodes in self.hidden_layer_conf:
                hidden_layers = tf.keras.layers.Dense(units=num_hidden_nodes,
                                                      activation = tf.keras.activations.sigmoid,
                                                      use_bias=True)(hidden_layers)

        output = tf.keras.layers.Dense(units=self.num_output_nodes,
                                       activation = tf.keras.activations.sigmoid,
                                       use_bias=True)(hidden_layers)

        self.logic_op_model = tf.keras.Model(inputs=input_layer, outputs=output)

        sgd = tf.keras.optimizers.SGD(learning_rate=0.1)
        self.logic_op_model.compile(optimizer=sgd, loss="mse")

    def fit(self, x, y, batch_size, epochs):
        self.logic_op_model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs)

    def predict(self, x, batch_size):
        prediction = self.logic_op_model.predict(x=x, batch_size=batch_size)
        return prediction