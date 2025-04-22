import tensorflow as tf
from MNISTData import MNISTData
import numpy as np

class CNN:
    def __init__(self, hidden_layer_conf, num_output_nodes):
        self.hidden_layer_conf = hidden_layer_conf
        self.num_output_nodes = num_output_nodes
        self.logic_op_model = None
        self.image_shape_x = None
        self.image_shape_y = None
        self.num_labels = None

    # CNN 
    def build_CNN_model(self):
        input_layer = tf.keras.Input(shape=[self.image_shape_x, self.image_shape_y, 1, ])

        # Conv + Pooling
        hidden_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), 
                                              padding="valid", activation='relu')(input_layer)
        hidden_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(hidden_layer)

        # Conv + Pooling
        hidden_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), 
                                              padding="valid", activation='relu')(hidden_layer)
        hidden_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(hidden_layer)

        # Flatten
        hidden_layer = tf.keras.layers.Flatten()(hidden_layer)

        # MLP
        hidden_layer = tf.keras.layers.Dense(units=64, activation='relu')(hidden_layer)
        output = tf.keras.layers.Dense(units=self.num_labels, activation='softmax')(hidden_layer)

        classifier_model = tf.keras.Model(inputs=input_layer, outputs=output)
        classifier_model.summary()

        opt_alg = tf.keras.optimizers.Adam(learning_rate = 0.001)
        loss_cross_e = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        classifier_model.compile(optimizer=opt_alg, loss=loss_cross_e, metrics=['accuracy'])
        self.classifier = classifier_model

    def fit(self, x, y, batch_size, epochs):
        self.logic_op_model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs)

    def predict(self, x, batch_size):
        prediction = self.logic_op_model.predict(x=x, batch_size=batch_size)
        return prediction