import os
import numpy as np
import Data_Utils
from warnings import simplefilter
from sklearn.preprocessing import StandardScaler, normalize
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold

# Kernel Setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Comment this line on other OS'
simplefilter(action="ignore", category=FutureWarning)
np.random.seed(123)

# Load Dataset
CU_X, Y = Data_Utils.get_text_dataset('datasets/casis25_bow.txt')
# CU_X, Y = Data_Utils.get_text_dataset('datasets/casis25_ncu.txt')
# CU_X, Y = Data_Utils.get_text_dataset('datasets/casis25_sty.txt')
# CU_X, Y = Data_Utils.get_text_dataset('datasets/casis25_char-gram_gram=3-limit=1000.txt')

rbfsvm = svm.SVC()
lsvm = svm.LinearSVC()
mlp = MLPClassifier(max_iter=2000)

skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
fold_accuracy = []

scaler = StandardScaler()
tfidf = TfidfTransformer(norm=None)
dense = Data_Utils.DenseTransformer()
iter_count = 0
for train, test in skf.split(CU_X, Y):
    iter_count += 1
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
    rbfsvm.fit(train_data, train_labels)
    lsvm.fit(train_data, train_labels)
    mlp.fit(train_data, train_labels)

    rbfsvm_acc = rbfsvm.score(eval_data, eval_labels)
    lsvm_acc = lsvm.score(eval_data, eval_labels)
    mlp_acc = mlp.score(eval_data, eval_labels)

    fold_accuracy.append((lsvm_acc, rbfsvm_acc, mlp_acc))
    print("Accuracy at iteration: "+str(iter_count)+"\n"+str((lsvm_acc, rbfsvm_acc, mlp_acc)))

print("\nMean accuracy: ")
print(np.mean(fold_accuracy, axis=0))