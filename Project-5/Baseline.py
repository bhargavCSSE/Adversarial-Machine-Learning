# -*- coding: utf-8 -*-
"""
@author: Bhargav Joshi
"""
import os
import Data_Utils
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from warnings import simplefilter

# Kernel Setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Comment this line on other OS'
simplefilter(action="ignore", category=FutureWarning)
np.random.seed(123)

# Load Data
# CU_X, raw_Y = Data_Utils.get_text_dataset('datasets/casis25_bow.txt')
CU_X, raw_Y = Data_Utils.get_text_dataset('datasets/casis25_ncu.txt')
# CU_X, raw_Y = Data_Utils.get_text_dataset('datasets/casis25_sty.txt')
# CU_X, raw_Y = Data_Utils.get_text_dataset('datasets/casis25_char-gram_gram=3-limit=1000.txt')

# Process Labels
Y = []
for elem in raw_Y:
    Y.append(int(elem) - 1000)
Y = np.array(Y)
Y = Y.astype(str)

# Define Model Architecture
NeuralNetwork = tf.keras.Sequential()
NeuralNetwork.add(tf.keras.layers.Flatten())
NeuralNetwork.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))
NeuralNetwork.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))
NeuralNetwork.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
NeuralNetwork.add(tf.keras.layers.Dense(25, activation=tf.nn.softmax))

# Baseline processing
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
fold_accuracy = []

scaler = StandardScaler()
tfidf = TfidfTransformer(norm=None)
dense = Data_Utils.DenseTransformer()

for train, test in skf.split(CU_X, Y):
    # train split
    CU_train_data = CU_X[train]
    train_labels = Y[train]

    # test split
    CU_eval_data = CU_X[test]
    eval_labels = Y[test]

    # tf-idf
    tfidf.fit(CU_train_data)
    CU_train_data = dense.transform(tfidf.transform(CU_train_data))
    CU_eval_data = dense.transform(tfidf.transform(CU_eval_data))

    # standardization
    scaler.fit(CU_train_data)
    CU_train_data = scaler.transform(CU_train_data)
    CU_eval_data = scaler.transform(CU_eval_data)

    # normalization
    CU_train_data = normalize(CU_train_data)
    CU_eval_data = normalize(CU_eval_data)

    train_data = CU_train_data
    eval_data = CU_eval_data

    # Train Model
    NeuralNetwork.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    NeuralNetwork.fit(train_data, train_labels, epochs=3)

    # Evaluate Model
    val_loss, val_acc = NeuralNetwork.evaluate(eval_data, eval_labels)
    pred_labels = NeuralNetwork.predict(eval_data)
    print("Test loss:" + str(val_loss) + "\n" + "Test acc:" + str(val_acc))

    fold_accuracy.append(val_acc)
print(np.mean(fold_accuracy, axis=0))
