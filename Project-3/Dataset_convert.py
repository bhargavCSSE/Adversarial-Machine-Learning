#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 18:09:20 2019

@author: bhargavjoshi
"""
import pandas as pd

file = open("Project3_Dataset_v1.txt","r")
lines = file.read().splitlines()
col_1 = []
col_2 = []
col_3 = []
for element in lines:
    col_1.append(float(element.split(' ')[0]))
    col_2.append(float(element.split(' ')[1]))
    col_3.append(float(element.split(' ')[2]))
file.close()

new_col_3 = [-1.0 if x<= 0.5 else 1.0 for x in col_3]
size = len(new_col_3)

df = pd.DataFrame(index = [i for i in range(size)], columns = ["Input_1", "Input_2", "Output"])
for i in range(size):
    df.iloc[i]["Input_1"] = col_1[i]
    df.iloc[i]["Input_2"] = col_2[i]
    df.iloc[i]["Output"] = new_col_3[i]

print(df)
df.to_csv("Project3_Dataset_v2.txt", sep=' ', header=None, index=False)