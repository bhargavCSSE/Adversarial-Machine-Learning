#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 18:09:20 2019

@author: bhargavjoshi
"""

#Library imports
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Tools import
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


class aFFNeuralNetwork:
    def __init__(self, filename):
        self.filename = filename
        self.X_train = []
        self.X_eval = []
        self.X_valid = []
        self.X_test = []
        self.y_train = []
        self.y_eval = []
        self.y_valid = []
        self.y_test = []
        self.pred_train = []
        self.pred_test = []
        
    def readData(self):
        dataset = pd.read_csv(self.filename, sep=' ', header=None)
        dataset = dataset.sample(n=len(dataset), random_state=42)
        DS_array = dataset.to_numpy()
        
        train = math.floor(0.7 * len(dataset))
        eval = math.floor(.8 * len(dataset))
        test = math.floor(.9 * len(dataset))
        
        X_train = DS_array[:train,0:2]
        y_train= DS_array[:train,2]
        
        X_eval = DS_array[train:eval,0:2]
        y_eval = DS_array[train:eval,2]
        
        X_valid = DS_array[eval:test,0:2]
        y_valid = DS_array[eval:test,2]
        
        X_test = DS_array[test:,0:2]
        y_test = DS_array[test:,2]
        
        return X_train, y_train, X_eval, y_eval, X_valid, y_valid, X_test, y_test
    
    def preProcessing(self, X, y, trainSize, testSize, scale):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = trainSize, test_size = testSize)
        if scale == True:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_scale = scaler.transform(X_train)
            X_test_scale = scaler.transform(X_test)
            return X_train_scale, X_test_scale, y_train, y_test
        elif scale == False:
            return X_train, X_test, y_train, y_test
        
    def DeployNeuralNet(self, solver, hiddenLayers, alpha = 0.001, iterations = 500):
        self.X_train, self.y_train, self.X_eval, self.y_eval, self.X_valid, self.y_valid, self.X_test, self.y_test = self.readData()
        #self.X_train, self.X_test, self.y_train, self.y_test = self.preProcessing(X, y, 0.7, 0.3, scale = False) 
        reg = MLPRegressor(solver=solver, hidden_layer_sizes=hiddenLayers, alpha=alpha, max_iter=iterations)
        reg.fit(self.X_train,self.y_train)
        self.pred_train = reg.predict(self.X_train)
        self.pred_valid = reg.predict(self.X_valid)
        self.pred_test = reg.predict(self.X_test)
        
    def Plotter(self, x, y, z, title = "title"):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1, projection='3d')
        ax1.scatter(x, y, z)
        plt.title(title)
        ax1.set_xlim3d(-100.0, 100.0)
        ax1.set_ylim3d(-100.0, 100.0)
        ax1.set_zlim3d(0, 1.0)
        plt.show()

    def PerformanceMatrix(self, X, y_actual, y_pred, plot = 1):
        tp = 0
        tn = 0
        fp = 0
        fn = 0  
        for test_instance_result, label in zip(y_pred, y_actual):
    
            if ((test_instance_result > 0.5) and (label > 0.5)):
                tp += 1
            if ((test_instance_result <= 0.5) and (label <= 0.5)):
                tn += 1
            if ((test_instance_result > 0.5) and (label <= 0.5)):
                fp += 1
            if ((test_instance_result <= 0.5) and (label > 0.5)):
                fn += 1

        MSE_train = mean_squared_error(self.y_train, self.pred_train)
        MSE_test = mean_squared_error(y_actual, y_pred)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        recall = tp / (tp + fn + 0.00001)
        precision = tp / (tp + fp + 0.00001)
        f1 = 2 * (precision * recall) / (precision + recall + 0.00001)
        print("Training MSE: ", MSE_train)
        print("Testing MSE:  ", MSE_test)
        print("Accuracy:     ", accuracy)
        print("Recall:       ", recall)
        print("Precision:    ", precision)
        print("F1:           ", f1)
        if plot == 1:
            self.Plotter(X[:,0], X[:,1], y_actual, "Actual")
            self.Plotter(X[:,0], X[:,1], y_pred, "Predicted")

FFNN = aFFNeuralNetwork("Project3_Dataset_v1.txt")
FFNN.DeployNeuralNet(solver = 'lbfgs', hiddenLayers=(15, 10, 10, 10), alpha=0.2, iterations=20000)
print("Validation Set")
FFNN.PerformanceMatrix(X=FFNN.X_valid ,y_pred=FFNN.pred_valid, y_actual= FFNN.y_valid, plot=0)
print("\nTest Set")
FFNN.PerformanceMatrix(X=FFNN.X_test ,y_pred=FFNN.pred_test, y_actual= FFNN.y_test, plot=0)