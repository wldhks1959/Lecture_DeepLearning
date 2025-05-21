# 학습한 모델로서 AE_test.py에서 사용할 수 있어야 한다. 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from AutoEncoder import AutoEncoder
from MNISTData import MNISTData

# noise 추가 
def noise_adder(x, drop_prob=0.5):
    noise = np.random.binomial(n=1, p=1-drop_prob, size=x.shape)
    return x*noise

# data loading and preprocessing
data = MNISTData()
data.load_data()

# AutoEncoder Init and Model Build
model = AutoEncoder()
model.input_output_dim = data.in_out_dim
model.build_model()

# create noise data
x_train_noised = noise_adder(data.x_train, 0.5)

# train
model.fit(x=x_train_noised, y=data.x_train, batch_size=128, epochs=20)

# save model
model.save_weights("model/autoEncoder.weights.h5")