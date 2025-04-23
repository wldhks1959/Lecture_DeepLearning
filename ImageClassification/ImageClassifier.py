import tensorflow as tf
import numpy as np

class ImageClassifier_MLP:
    def __init__(self, img_shape_x, img_shape_y, num_labels):
        self.img_shape_x = img_shape_x
        self.img_shape_y = img_shape_y
        self.num_labels = num_labels
        self.classifier = None
    
    def fit(self, train_imgs, train_labels, num_epochs):
        self.classifier.fit(train_imgs, train_labels, epochs=num_epochs)

    def predict(self, test_imgs):
        predictions = self.classifier.predict(test_imgs)
        return predictions
