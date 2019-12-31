# -*- coding: utf-8 -*-
"""
@author: Armin Khayyer
"""
import os
import random
import sys
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Data_Utils as Data_Utils
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


def unigram():
    my_zip = zipfile.ZipFile("CASIS-25_Dataset.zip")
    all_files = my_zip.namelist()
    all_files = sorted(all_files)
    look_up_table = sorted([i for i in string.printable])
    look_up_table = look_up_table[5:]
    feature_df = pd.DataFrame(columns=look_up_table)

    for file in all_files:
        feature_df.loc[file] = 0
        f = my_zip.open(file)
        for line in f:
            for ch in line:
                char = chr(ch)
                if char in look_up_table:
                    feature_df.loc[file, char] += 1

    feature_df = feature_df.reset_index()
    feature_df["label"] = feature_df.apply(lambda row: row["index"][:4], 1)
    feature_df = feature_df.drop(["index"], 1)
    return feature_df

try:
    feature_df = pd.read_csv("features.csv", sep=",")
except:
    feature_df = unigram()
    feature_df.to_csv("features.csv", sep=",", index=False)


def feature_selection(mask):
    df_x = feature_df.drop(["label"], 1)
    df_x = df_x.loc[:, mask]
    x = np.array(df_x)
    y = np.array(feature_df["label"])
    return x, y

def Baselin(mask):
    CU_X, Y = feature_selection(mask)

    rbfsvm = svm.SVC()
    lsvm = svm.LinearSVC()
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
        rbfsvm.fit(train_data, train_labels)
        lsvm.fit(train_data, train_labels)
        mlp.fit(train_data, train_labels)

        rbfsvm_acc = rbfsvm.score(eval_data, eval_labels)
        lsvm_acc = lsvm.score(eval_data, eval_labels)
        mlp_acc = mlp.score(eval_data, eval_labels)

        fold_accuracy.append((lsvm_acc, rbfsvm_acc, mlp_acc))
    return (np.mean(fold_accuracy, axis=0))


class anIndividual:
    def __init__(self, specified_chromosome_length):
        self.chromosome = []
        self.fitness_RBFSVM = 0
        self.fitness_LSVM = 0
        self.fitness_MLP = 0
        self.fitness = 0
        self.chromosome_length = specified_chromosome_length
        
    def randomly_generate(self):
        for i in range(self.chromosome_length):
            self.chromosome.append(random.choice([True, False]))
    
    def calculate_fitness(self):
        self.fitness_RBFSVM, self.fitness_LSVM, self.fitness_MLP = Baselin(self.chromosome)
        self.fitness = (self.fitness_RBFSVM +  self.fitness_LSVM + self.fitness_MLP)/3


    def print_individual(self, i):
        print("Chromosome - "+str(i) +"- number of features: " + str(sum(self.chromosome)) + " Fitness: " + str(self.fitness), " Fitness-tuple: " + str(self.fitness_RBFSVM),  str(self.fitness_LSVM),str(self.fitness_MLP) )
      
class aSimpleExploratoryAttacker:
    def __init__(self, population_size, chromosome_length, mutation_rate):
        if (population_size < 2):
            print("Error: Population Size must be greater than 2")
            sys.exit()
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.mutation_amt = mutation_rate
        self.population = []
        
    def generate_initial_population(self):
        for i in range(self.population_size):
            individual = anIndividual(self.chromosome_length)
            individual.randomly_generate()
            individual.calculate_fitness()
            self.population.append(individual)

    def get_worst_fit_individual(self):
        worst_fitness = 999999999.0  # For Maximization
        worst_individual = -1
        for i in range(self.population_size):
            if (self.population[i].fitness < worst_fitness):
                worst_fitness = self.population[i].fitness
                worst_individual = i
            elif (self.population[i].fitness == worst_fitness):
                if sum(self.population[i].chromosome) > sum(self.population[worst_individual].chromosome):
                    worst_fitness = self.population[i].fitness
                    worst_individual = i
        return worst_individual



    # def get_worst_fit_individual(self):
    #     worst_fitness = 999999999.0  # For Maximization
    #     worst_individual = -1
    #     for i in range(self.population_size):
    #         if (self.population[i].fitness < worst_fitness):
    #             worst_fitness = self.population[i].fitness
    #             worst_individual = i
    #     return worst_individual
    
    def get_best_fitness(self):
        best_fitness = -99999999999.0
        best_individual = -1
        for i in range(self.population_size):
            if self.population[i].fitness > best_fitness:
                best_fitness = self.population[i].fitness
                best_individual = i
        return best_fitness


    # def tournoment_selection(self, k=2):
    #     tournoment_output1 = random.choices(self.population, k=k)
    #     best_indivisual1 = [i.fitness for i in tournoment_output1]
    #     parent1 = tournoment_output1[best_indivisual1.index(max(best_indivisual1))]
    #     tournoment_output2 = random.choices(self.population, k=k)
    #     best_indivisual2 = [i.fitness for i in tournoment_output2]
    #     parent2 = tournoment_output2[best_indivisual2.index(max(best_indivisual2))]
    #     return parent1, parent2



    def tournoment_selection(self, k=2):
        tournoment_output1 = random.choices(self.population, k=k)
        best_indivisual1 = [i.fitness for i in tournoment_output1]
        if best_indivisual1[0] > best_indivisual1[1]:
            parent1 = tournoment_output1[0]
        elif best_indivisual1[0] == best_indivisual1[1]:
            if sum(tournoment_output1[0].chromosome) < sum(tournoment_output1[1].chromosome):
                parent1 = tournoment_output1[0]
            else:
                parent1 = tournoment_output1[1]
        else:
            parent1 = tournoment_output1[1]

        #parent1 = tournoment_output1[best_indivisual1.index(max(best_indivisual1))]

        tournoment_output2 = random.choices(self.population, k=k)
        best_indivisual2 = [i.fitness for i in tournoment_output2]
        if best_indivisual2[0] > best_indivisual2[1]:
            parent2 = tournoment_output2[0]
        elif best_indivisual2[0] == best_indivisual2[1]:
            if sum(tournoment_output2[0].chromosome) < sum(tournoment_output2[1].chromosome):
                parent2 = tournoment_output2[0]
            else:
                parent2 = tournoment_output2[1]
        else:
            parent2 = tournoment_output2[1]
        # parent2 = tournoment_output2[best_indivisual2.index(max(best_indivisual2))]
        return parent1, parent2



    def Crossover_operator(self, mom, dad):
        kid = anIndividual(self.chromosome_length)
        kid.randomly_generate()
        for j in range(self.chromosome_length):
            prob = random.uniform(0, 1)
            prob_mut = random.uniform(0, 1)
            if prob <= .5:
                kid.chromosome[j] = mom.chromosome[j]
            else:
                kid.chromosome[j] = dad.chromosome[j]

            if prob_mut >= self.mutation_amt:
                pass
            else:
                kid.chromosome[j] = not kid.chromosome[j]
        return kid


        
    def evolutionary_cycle(self):
        mom, dad = self.tournoment_selection()
        worst_individual = self.get_worst_fit_individual()
        self.population.pop(worst_individual)
        kid = self.Crossover_operator(mom, dad)
        self.population.append(kid)
        kid.calculate_fitness()

       
    def print_population(self):
        for i in range(self.population_size):
            self.population[i].print_individual(i)
    
    def print_best_max_fitness(self):
        best_fitness = -999999999.0  # For Maximization
        best_individual = -1
        for i in range(self.population_size):
            if self.population[i].fitness > best_fitness:
                best_fitness = self.population[i].fitness
                best_individual = i
        print("Best Indvidual: ",str(best_individual)," ", self.population[best_individual].chromosome, " Fitness: ", str(best_fitness))
        return self.population[best_individual]

    # def plot_evolved_candidate_solutions(self):
    #     fig = plt.figure()
    #     ax1 = fig.add_subplot(1,1,1,projection='3d')
    #     ax1.scatter(self.hacker_tracker_x,self.hacker_tracker_y,self.hacker_tracker_z)
    #     plt.title("Evolved Candidate Solutions")
    #     ax1.set_xlim3d(-100.0,100.0)
    #     ax1.set_ylim3d(-100.0,100.0)
    #     ax1.set_zlim3d(0.2,1.0)
    #     plt.show()


simplefilter(action='ignore', category=FutureWarning)
look_up_table = sorted([i for i in string.printable])
look_up_table = look_up_table[5:]
look_up_table = look_up_table + ["fitness", "fitness_lsvm", "fitness_RBFSVM", "fitness_MLP"]
results = pd.DataFrame(columns=look_up_table)

for rep in range(10):
    ChromLength = 95
    MaxEvaluations = 1000

    PopSize = 50
    mu_amt  = 0.01

    simple_exploratory_attacker = aSimpleExploratoryAttacker(chromosome_length=ChromLength, mutation_rate=mu_amt, population_size=PopSize)

    simple_exploratory_attacker.generate_initial_population()
    simple_exploratory_attacker.print_population()
    best = 0
    for i in range(MaxEvaluations-PopSize):
        best = i
        simple_exploratory_attacker.evolutionary_cycle()
        if (i % PopSize == 0):
            print("At Iteration: " + str(i))
            simple_exploratory_attacker.print_population()

    print("\nFinal Population\n")
    simple_exploratory_attacker.print_population()
    best_indiv = simple_exploratory_attacker.print_best_max_fitness()
    print("Function Evaluations: " + str(i))
    # simple_exploratory_attacker.plot_evolved_candidate_solutions()
    row = best_indiv.chromosome + [best_indiv.fitness, best_indiv.fitness_LSVM, best_indiv.fitness_RBFSVM, best_indiv.fitness_MLP]
    results.loc[rep, :] = row

print(results)
results.to_csv("final_results.csv", sep=",")

