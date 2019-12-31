import os
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from matplotlib import pyplot as plt
from warnings import simplefilter

# Kernel Setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Comment this line on other OS'
simplefilter(action="ignore", category=FutureWarning)
np.random.seed(123)

# Load Data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
plt.imshow(X_train[0])

# Pre-processing data
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# Define Model Architecture
NeuralNetwork = tf.keras.Sequential()
NeuralNetwork.add(tf.keras.layers.Flatten())
NeuralNetwork.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
NeuralNetwork.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
NeuralNetwork.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# Train Model
NeuralNetwork.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
NeuralNetwork.fit(X_train, y_train, epochs=3)

# Test Model
y_pred = NeuralNetwork.predict(X_train)
val_loss, val_acc = NeuralNetwork.evaluate(X_test, y_test)
print("Test loss:" + str(val_loss) + "   " + "Test acc:" + str(val_acc))
print(np.argmax(y_pred[0]))