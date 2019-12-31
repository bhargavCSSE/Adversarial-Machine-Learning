#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:07:36 2019

@author: Armin Khayyer, Bhargav Joshi, Ye Wang

@Base code provided by Gerry Dozier
"""

import os
import random
import sys
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

class aGaussianKernel:
    def __init__(self, target, desired_output):
        self.target = target
        self.desired_output = desired_output

    def fire_strength(self, q, sigma):
        sum_squared = 0.0
        for i in range(0, len(q)):
            sum_squared += math.pow((q[i] - self.target[i]), 2.0)
        # print("FS Sum_Squared = " + str(sum_squared))
        return math.exp(-sum_squared / (2.0 * pow(sigma, 2.0)))

    def print_gaussian_kernel(self):
        print("Target = ", self.target, " Desired Output = " + str(self.desired_output))


class aSimpleNeuralNetwork:
    def __init__(self, dataset_file_name, num_of_inputs, num_of_outputs):
        self.filename = dataset_file_name
        self.num_of_inputs = num_of_inputs
        self.num_of_outputs = num_of_outputs
        self.neuron = []
        self.sigma = 0
        self.tp = 0  # true positive
        self.tn = 0  # true negative
        self.fp = 0  # false positive
        self.fn = 0  # false negative

        self.tp_valid = 0  # true positive
        self.tn_valid = 0  # true negative
        self.fp_valid = 0  # false positive
        self.fn_valid = 0  # false negative

        self.tp_test = 0  # true positive
        self.tn_test = 0  # true negative
        self.fp_test = 0  # false positive
        self.fn_test = 0  # false negative




        df = pd.read_csv(self.filename, sep=" ", names=["x", "y", "z"])
        df = df.sample(n=len(df), random_state=42)
        train = math.floor(0.7 * len(df)) +1
        eval = math.floor(.8 * len(df)) +1
        test = math.floor(.9 * len(df)) +1
        df_train = df[:train]
        df_eval = df[train:eval]
        df_valid = df[eval:test]
        df_test = df[test:]

        self.training_instance = np.array(df_train)
        self.test_set = np.array(df_test)
        self.eval_set = np.array(df_eval)
        self.valid_set = np.array(df_valid)

        # with open(self.dataset_file_name, "r") as dataset_file:
        #     for line in dataset_file:
        #         line = line.strip().split(" ")
        #         self.training_instance.append([float(x) for x in line[0:]])

        for i in range(len(self.training_instance)):
            temp = aGaussianKernel(self.training_instance[i][:self.num_of_inputs],
                                   self.training_instance[i][self.num_of_inputs])
            self.neuron.append(temp)

    def set_sigma(self, sigma):
        self.sigma = sigma

    def distance_squared(self, x, y):
        dist_sqrd = 0
        for i in range(len(x)):
            dist_sqrd += (x[i] - y[i]) ** 2
        return dist_sqrd

    def train(self):
        dmax = 0
        dist_squared = 0
        for i in range(len(self.neuron) - 1):
            for j in range((i + 1), len(self.neuron)):
                dist_squared = self.distance_squared(self.neuron[i].target, self.neuron[j].target)
                if dmax < dist_squared:
                    dmax = dist_squared
        self.sigma = math.sqrt(dmax)
        print("dmax =", self.sigma)

    def check(self, query):
        sum_fire_strength = 0
        sum_fire_strength_x_desired_output = 0
        for i in range(len(self.neuron)):
            the_fire_strength = self.neuron[i].fire_strength(query, self.sigma)
            sum_fire_strength_x_desired_output += the_fire_strength * self.neuron[i].desired_output
            sum_fire_strength += the_fire_strength
        if (sum_fire_strength == 0.0):
            sum_fire_strength = 0.000000001  # to prevent divide by zero
        return sum_fire_strength_x_desired_output / sum_fire_strength

    def test_model(self):
        sum_squared_error = 0
        for test_case in self.eval_set:
            test_instance_result = self.check(test_case[:2])
            sum_squared_error += (test_instance_result - test_case[2]) ** 2
            self.calculate_statistics(test_instance_result, test_case[2])
        return (sum_squared_error / len(self.eval_set))


    def test_model_validation(self):
        sum_squared_error = 0
        for test_case in self.valid_set:
            test_instance_result = self.check(test_case[:2])
            sum_squared_error += (test_instance_result - test_case[2]) ** 2
            sum_squared_error = (sum_squared_error / len(self.valid_set))
            if ((test_instance_result > 0.5) and (test_case[2] > 0.5)):
                self.tp_valid += 1
            if ((test_instance_result <= 0.5) and (test_case[2] <= 0.5)):
                self.tn_valid += 1
            if ((test_instance_result > 0.5) and (test_case[2] <= 0.5)):
                self.fp_valid += 1
            if ((test_instance_result <= 0.5) and (test_case[2] > 0.5)):
                self.fn_valid += 1
        accuracy = (self.tp_valid + self.tn_valid) / (self.tp_valid+ self.tn_valid + self.fp_valid + self.fn_valid)
        recall = self.tp_valid / (self.tp_valid + self.fn_valid+ 0.00001)
        precision = self.tp_valid / (self.tp_valid + self.fp_valid + 0.00001)
        f1 = 2 * (precision * recall) / (precision + recall + 0.00001)
        return accuracy

    def test_model_test(self):
        sum_squared_error = 0
        for test_case in self.test_set:
            test_instance_result = self.check(test_case[:2])
            sum_squared_error += (test_instance_result - test_case[2]) ** 2
            if ((test_instance_result > 0.5) and (test_case[2] > 0.5)):
                self.tp_test += 1
            if ((test_instance_result <= 0.5) and (test_case[2] <= 0.5)):
                self.tn_test += 1
            if ((test_instance_result > 0.5) and (test_case[2] <= 0.5)):
                self.fp_test += 1
            if ((test_instance_result <= 0.5) and (test_case[2] > 0.5)):
                self.fn_test += 1
        sum_squared_error = (sum_squared_error / len(self.test_set))
        print("mse:", sum_squared_error)
        accuracy = (self.tp_test + self.tn_test) / (self.tp_test + self.tn_test + self.fp_test + self.fn_test)
        recall = self.tp_test / (self.tp_test + self.fn_test + 0.00001)
        precision = self.tp_test / (self.tp_test + self.fp_test + 0.00001)
        f1 = 2 * (precision * recall) / (precision + recall + 0.00001)
        print("Accuracy:  ", accuracy)
        print("Recall:    ", recall)
        print("Precision: ", precision)
        print("F1:        ", f1)
        print("sigma", self.sigma)
        return accuracy

    def calculate_statistics(self, test_instance_result, SchafferF6):
        # print("in calculate", test_instance_result, " ", SchafferF6)
        if ((test_instance_result > 0.5) and (SchafferF6 > 0.5)):
            self.tp += 1
        if ((test_instance_result <= 0.5) and (SchafferF6 <= 0.5)):
            self.tn += 1
        if ((test_instance_result > 0.5) and (SchafferF6 <= 0.5)):
            self.fp += 1
        if ((test_instance_result <= 0.5) and (SchafferF6 > 0.5)):
            self.fn += 1

    # def plot_model(self, number_of_test_cases, lb, ub):
    #     test_case_x = []
    #     test_case_y = []
    #     test_case_z = []
    #
    #     schafferF6_x = []
    #     schafferF6_y = []
    #     schafferF6_z = []
    #
    #     for i in range(number_of_test_cases):
    #         x = random.uniform(lb, ub)
    #         y = random.uniform(lb, ub)
    #
    #         test_case_x.append(x)
    #         test_case_y.append(y)
    #         schafferF6_x.append(x)
    #         schafferF6_y.append(y)
    #
    #         test_case = []
    #         test_case.append(x)
    #         test_case.append(y)
    #         # print("test case: ",test_case)
    #         test_case_z.append(self.check(test_case))
    #
    #         x2y2 = x ** 2 + y ** 2
    #         SchafferF6 = 0.5 + (math.sin(math.sqrt(x2y2)) ** 2 - 0.5) / (1 + 0.001 * x2y2) ** 2
    #         schafferF6_z.append(SchafferF6)
    #
    #     fig = plt.figure()
    #     ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    #     ax1.scatter(test_case_x, test_case_y, test_case_z)
    #     plt.title("Simple Neural Network")
    #     ax1.set_zlim3d(0.2, 1.0)
    #     plt.show()
    #
    #     fig = plt.figure()
    #     ax2 = fig.add_subplot(1, 1, 1, projection='3d')
    #     ax2.scatter(schafferF6_x, schafferF6_y, schafferF6_z)
    #     plt.title("SchafferF6")
    #     ax2.set_zlim3d(0.2, 1.0)
    #     plt.show()

    def print_training_set(self):
        print(self.training_instance)
        print(len(self.training_instance))

    def print_neurons(self):
        for i in range(len(self.neuron)):
            self.neuron[i].print_gaussian_kernel()

    def print_statistics(self):
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
        recall = self.tp / (self.tp + self.fn + 0.00001)
        precision = self.tp / (self.tp + self.fp + 0.00001)
        f1 = 2 * (precision * recall) / (precision + recall + 0.00001)
        print("Accuracy:  ", accuracy)
        print("Recall:    ", recall)
        print("Precision: ", precision)
        print("F1:        ", f1)
        return accuracy


class anIndividual:
    def __init__(self, specified_chromosome_length):
        self.chromosome = []
        self.fitness = 0
        self.fitness_test = 0
        self.fitness_validation = 0
        self.mse = 0
        self.chromosome_length = specified_chromosome_length


    def randomly_generate(self, lb, ub):
        for i in range(self.chromosome_length):
            self.chromosome.append(random.uniform(lb, ub))

    def calculate_fitness(self):
        simple_neural_network = aSimpleNeuralNetwork("Project3_Dataset_v1.txt", 2, 1)
        simple_neural_network.train()
        simple_neural_network.set_sigma(self.chromosome[0])
        self.mse = simple_neural_network.test_model()
        print("Model Test AMSE: ", self.mse )
        self.fitness = simple_neural_network.print_statistics()
        self.fitness_validation = simple_neural_network.test_model_validation()
        self.fitness_test = simple_neural_network.test_model_test()


    def print_individual(self, i):
        print("Chromosome " + str(i) + ": " + str(self.chromosome) + " Fitness: " + str(self.fitness), " Fitness validation: " + str(self.fitness_validation) )


class aSimpleExploratoryAttacker:
    def __init__(self, population_size, chromosome_length, mutation_rate, lb, ub ):
        if (population_size < 2):
            print("Error: Population Size must be greater than 2")
            sys.exit()
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.mutation_amt = mutation_rate
        self.lb = lb
        self.ub = ub
        self.mutation_amt = mutation_rate * (ub - lb)
        self.population = []
        self.hacker_tracker_x = []
        self.hacker_tracker_z = []
        self.best_fit_validation = ""


    def generate_initial_population(self):
        for i in range(self.population_size):
            individual = anIndividual(self.chromosome_length)
            individual.randomly_generate(self.lb, self.ub)
            individual.calculate_fitness()
            self.hacker_tracker_x.append(individual.chromosome[0])
            self.hacker_tracker_z.append(individual.fitness)
            self.population.append(individual)
        self.best_fit_validation = self.get_best_fit_individual_validation()

    def get_worst_fit_individual(self):
        worst_fitness = 999999999.0  # For Maximization
        worst_individual = -1
        for i in range(self.population_size):
            if (self.population[i].fitness < worst_fitness):
                worst_fitness = self.population[i].fitness
                worst_individual = i
        return worst_individual

    def get_best_fitness(self):
        best_fitness = -99999999999.0
        best_individual = -1
        for i in range(self.population_size):
            if self.population[i].fitness > best_fitness:
                best_fitness = self.population[i].fitness
                best_individual = i
        return best_fitness

    def get_best_fit_individual(self):
        best_fitness = -99999999999.0
        best_individual = -1
        for i in range(self.population_size):
            if self.population[i].fitness > best_fitness:
                best_fitness = self.population[i].fitness
                best_individual = i
        return self.population[best_individual]



    def get_best_fit_individual_validation(self):
        best_fitness = -99999999999.0
        best_individual = -1
        for i in range(self.population_size):
            if self.population[i].fitness_validation > best_fitness:
                best_fitness = self.population[i].fitness_validation
                best_individual = i
        return self.population[best_individual]



    def tournoment_selection(self, k=2):
        tournoment_output1 = random.choices(self.population, k=k)
        best_indivisual1 = [i.fitness for i in tournoment_output1]
        parent1 = tournoment_output1[best_indivisual1.index(max(best_indivisual1))]
        tournoment_output2 = random.choices(self.population, k=k)
        best_indivisual2 = [i.fitness for i in tournoment_output2]
        parent2 = tournoment_output2[best_indivisual2.index(max(best_indivisual2))]
        return parent1, parent2

    def Crossover_operator(self, mom, dad):
        kid = anIndividual(self.chromosome_length)
        kid.randomly_generate(self.lb, self.ub)
        for j in range(self.chromosome_length):
            mean = (mom.chromosome[j] + dad.chromosome[j])/2
            std = np.std(np.array([mom.chromosome[j],dad.chromosome[j] ]))
            kid.chromosome[j] = mean + std * random.normalvariate(0, 1)
            if kid.chromosome[j] > self.ub:
                kid.chromosome[j] = self.ub
            if kid.chromosome[j] < self.lb:
                kid.chromosome[j] = self.lb
        return kid


    def evolutionary_cycle(self):
        mom, dad = self.tournoment_selection()
        worst_individual = self.get_worst_fit_individual()
        best_individual = self.get_best_fit_individual()
        self.population.pop(worst_individual)
        kid = self.Crossover_operator(mom, dad)
        self.population.append(kid)
        kid.calculate_fitness()
        self.hacker_tracker_x.append(kid.chromosome[0])
        self.hacker_tracker_z.append(kid.fitness)
        if kid.fitness > best_individual.fitness:
            if kid.fitness_validation > self.best_fit_validation.fitness_validation:
                self.best_fit_validation = kid




       
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
        print("Best Indvidual: ", str(best_individual), " ", self.population[best_individual].chromosome, " Fitness: ",
              str(best_fitness))
        return best_fitness

    def plot_evolved_candidate_solutions(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1,)
        ax1.scatter(self.hacker_tracker_x, self.hacker_tracker_z)
        plt.title("Evolved Candidate Solutions")
        # ax1.set_xlim(-100.0, 100.0)
        # ax1.set_zlim(0.2, 1.0)
        plt.show()


ChromLength = 1
ub = 100
lb = .1
MaxEvaluations = 500
plot = 0

PopSize = 50
mu_amt = 0.01

simple_exploratory_attacker = aSimpleExploratoryAttacker(PopSize, ChromLength, mu_amt, lb, ub)

simple_exploratory_attacker.generate_initial_population()
simple_exploratory_attacker.print_population()

for i in range(MaxEvaluations - PopSize + 1):
    simple_exploratory_attacker.evolutionary_cycle()
    if (i % PopSize == 0):
        if (plot == 1):
            simple_exploratory_attacker.plot_evolved_candidate_solutions()
        print("At Iteration: " + str(i))
        simple_exploratory_attacker.print_population()
    if (simple_exploratory_attacker.get_best_fitness() >= 0.99754):
        break

print("\nFinal Population\n")
simple_exploratory_attacker.print_population()
simple_exploratory_attacker.print_best_max_fitness()
print("Function Evaluations: " + str(PopSize + i))
simple_exploratory_attacker.plot_evolved_candidate_solutions()
print( "validation", simple_exploratory_attacker.best_fit_validation.chromosome, simple_exploratory_attacker.best_fit_validation.fitness_validation)
print( "test", simple_exploratory_attacker.best_fit_validation.chromosome, simple_exploratory_attacker.best_fit_validation.fitness_test)





simple_neural_network = aSimpleNeuralNetwork("Project3_Dataset_v1.txt", 2, 1)
simple_neural_network.train()
simple_neural_network.set_sigma(simple_exploratory_attacker.best_fit_validation.chromosome[0])
mse = simple_neural_network.test_model()
print("Model Test AMSE: ", mse )
fitness = simple_neural_network.print_statistics()
fitness_validation = simple_neural_network.test_model_validation()
print("test results")
fitness_test = simple_neural_network.test_model_test()