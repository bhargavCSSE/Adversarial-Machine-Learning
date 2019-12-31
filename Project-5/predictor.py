import os
import random
import sys
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler, normalize
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import cross_val_score
import zipfile
import string
import pandas as pd
from warnings import simplefilter
import pickle
from sklearn.model_selection import train_test_split
import Data_Utils
from Extractor.DatasetInfo import DatasetInfo
from Extractor.Extractors import BagOfWords, Stylomerty, Unigram, CharacterGram


mask = np.load("mask.npy")
df = pd.read_csv('data/train/casis25_ncu.txt', header=None)
features = ['casis25_char-gram_gram=3-limit=1000.txt', 'casis25_bow.txt', 'casis25_sty.txt']

for feature in features:
    df_feature = pd.read_csv("data/train/" + feature, header=None)
    df = pd.merge(df, df_feature, on=0, how="left")
    print(df_feature.shape)
    print('adding {}'.format(feature))
print(df)


df["label"] = df[0].map(lambda x: str(x)[0:4])
df = df.drop(df.columns[[0]], axis=1)
feature_df = df
df_x = feature_df.drop(["label"], 1)
df_x = df_x.loc[:, mask]
x = np.array(df_x)
y = np.array(feature_df["label"])

CU_X, Y = x, y

# rbfsvm = svm.SVC()
# lsvm = svm.LinearSVC()
mlp = MLPClassifier(max_iter=2000)

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

    # evaluation
    # rbfsvm.fit(train_data, train_labels)
    # lsvm.fit(train_data, train_labels)
    mlp.fit(train_data, train_labels)

    # rbfsvm_acc = rbfsvm.score(eval_data, eval_labels)
    # lsvm_acc = lsvm.score(eval_data, eval_labels)
    mlp_acc = mlp.score(eval_data, eval_labels)

    fold_accuracy.append(mlp_acc)
print(np.mean(fold_accuracy))


# df_test = pd.read_csv("data/AdversarialTests.txt", header = None)
data_dir = "data/AdversarialTest"
feature_set_dir = "./datasets/"
for i in range(4):
    if i == 0:
        extractor = Unigram(data_dir, "casis25_test")
    elif i == 1:
        extractor = Stylomerty(data_dir, "casis25_test")
    elif i == 2:
        extractor = BagOfWords(data_dir, "casis25_test")
    else:
        extractor = CharacterGram(data_dir, "casis25_test", gram=3, limit=1000)

    extractor.start()
    lookup_table = extractor.lookup_table
    print("Generated Lookup Table:")
    # print(lookup_table)
    col = []
    if lookup_table is not False:
        print("'" + "', '".join([str("".join(x)).replace("\n", " ") for x in lookup_table]) + "'")
        for x in lookup_table:
            col.append("'" + "', '".join([str("".join(x)).replace("\n", " ")]) + "'")
        generated_file = feature_set_dir + extractor.out_file + ".txt"
        generated_csv_file = feature_set_dir + extractor.out_file + "_test.csv"
        data, labels = Data_Utils.get_dataset(generated_file)
        df = pd.DataFrame(data, columns=col)
        df.insert(0, "Label", labels, True)
        df.to_csv(generated_csv_file)
    else:
        generated_file = feature_set_dir + extractor.out_file + ".txt"
        generated_csv_file = feature_set_dir + extractor.out_file + "_test.csv"
        data, labels = Data_Utils.get_dataset(generated_file)
        df = pd.DataFrame(data)
        df.insert(0, "Label", labels, True)
        df.to_csv(generated_csv_file)

    # Get dataset information
    dataset_info = DatasetInfo("casis25_bow")
    dataset_info.read()
    authors = dataset_info.authors
    writing_samples = dataset_info.instances
    print("\n\nAuthors in the dataset:")
    print(authors)

    print("\n\nWriting samples of an author 1000")
    print(authors["1000"])

    print("\n\nAll writing samples in the dataset")
    print(writing_samples)

    print("\n\nThe author of the writing sample 1000_1")
    print(writing_samples["1000_1"])

    # print(labels[0], data[0])
print("Done")

df_test = pd.read_csv('datasets/casis25_test_ncu.txt', header=None)
features_test = ['casis25_test_char-gram_gram=3-limit=1000.txt', 'casis25_test_bow.txt', 'casis25_test_sty.txt']

for feature in features_test:
    df_feature = pd.read_csv("datasets/" + feature, header=None)
    df_test = pd.merge(df_test, df_feature, on=0, how="left")
    print(df_feature.shape)
    print('adding {}'.format(feature))
print(df_test)



df_test["label"] = df_test[0].map(lambda x: str(x)[0:4])
df_test = df_test.drop(df_test.columns[[0]], axis=1)
feature_df = df_test
df_x = feature_df.drop(["label"], 1)
df_x = df_x.loc[:, mask]
x = np.array(df_x)
y = np.array(feature_df["label"])