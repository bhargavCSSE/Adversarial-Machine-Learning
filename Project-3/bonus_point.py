# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 19:31:41 2019

@author: Gerry Dozier
"""

import os
import random
import sys
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from scipy import stats

#
#  A Simple Steady-State, Real-Coded Genetic Algorithm       
#
allindividual = []
class anIndividual:
    def __init__(self, specified_chromosome_length):
        self.chromosome = []
        self.fitness    = 0
        self.chromosome_length = specified_chromosome_length

        
    def randomly_generate(self,lb, ub):
        for i in range(self.chromosome_length):
            self.chromosome.append(random.uniform(lb, ub))
        self.fitness = 0

    
    def calculate_fitness(self):
        x2y2 = self.chromosome[0]**2 + self.chromosome[1]**2
        self.fitness = 0.5 + (math.sin(math.sqrt(x2y2))**2 - 0.5) / (1+0.001*x2y2)**2

    def print_individual(self, i):
        print("Chromosome "+str(i) +": " + str(self.chromosome) + " Fitness: " + str(self.fitness))


      
class aSimpleExploratoryAttacker:
    def __init__(self, population_size, chromosome_length, mutation_rate, lb, ub):
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
        self.hacker_tracker_y = []
        self.hacker_tracker_z = []
        
    def generate_initial_population(self):
        for i in range(self.population_size):
            individual = anIndividual(self.chromosome_length)
            individual.randomly_generate(self.lb,self.ub)
            individual.calculate_fitness()
            allindividual.append(individual)
            self.hacker_tracker_x.append(individual.chromosome[0])
            self.hacker_tracker_y.append(individual.chromosome[1])
            self.hacker_tracker_z.append(individual.fitness)
            self.population.append(individual)
    
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
        
    def evolutionary_cycle(self):
        mom = random.randint(0,self.population_size-1)
        dad = random.randint(0,self.population_size-1)
        worst = self.get_worst_fit_individual()
        kid = anIndividual(self.chromosome_length)
        kid.randomly_generate(self.lb, self.ub)
        self.population.pop(worst)
        self.population.append(kid)
        kid = self.population.index(kid)
        for j in range(self.chromosome_length):
            self.population[kid].chromosome[j] = random.uniform(self.population[mom].chromosome[j],self.population[dad].chromosome[j])
            self.population[kid].chromosome[j] += self.mutation_amt * random.gauss(0,1.0)
            if self.population[kid].chromosome[j] > self.ub:
                self.population[kid].chromosome[j] = self.ub
            if self.population[kid].chromosome[j] < self.lb:
                self.population[kid].chromosome[j] = self.lb
        self.population[kid].calculate_fitness()
        allindividual.append(self.population[kid])
        self.hacker_tracker_x.append(self.population[kid].chromosome[0])
        self.hacker_tracker_y.append(self.population[kid].chromosome[1])
        self.hacker_tracker_z.append(self.population[kid].fitness)
       
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
    
    def plot_evolved_candidate_solutions(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1,projection='3d')
        ax1.scatter(self.hacker_tracker_x,self.hacker_tracker_y,self.hacker_tracker_z)
        plt.title("Evolved Candidate Solutions")
        ax1.set_xlim3d(-100.0,100.0)
        ax1.set_ylim3d(-100.0,100.0)
        ax1.set_zlim3d(0.2,1.0)
        plt.show()


ChromLength = 2
ub = 100.0
lb = -100.0
MaxEvaluations = 1000
plot = 0

PopSize = 5




pop_size = [50]#, 3, 12, 25, 50, 100]
muamt =  .01
for k in pop_size:
    simple_exploratory_attacker = aSimpleExploratoryAttacker(k, ChromLength, muamt, lb, ub)
    simple_exploratory_attacker.generate_initial_population()
    simple_exploratory_attacker.print_population()
    best = 0
    for i in range(MaxEvaluations-k+1):
        best = i
        simple_exploratory_attacker.evolutionary_cycle()
        if (i % k == 0):
            if (plot == 1):
                simple_exploratory_attacker.plot_evolved_candidate_solutions()
            print("At Iteration: " + str(i))
            simple_exploratory_attacker.print_population()
        # if (simple_exploratory_attacker.get_best_fitness() >= 0.99754):
        #     break


    print("\nFinal Population\n")
    simple_exploratory_attacker.print_population()
    simple_exploratory_attacker.print_best_max_fitness()
    print("Function Evaluations: " + str(k+i))
    simple_exploratory_attacker.plot_evolved_candidate_solutions()
    print("best :", i)


data = [[i.chromosome[0],i.chromosome[1] , i.fitness] for i in allindividual]
df = pd.DataFrame(data, columns=["x", "y", "z"])
df_given = pd.read_csv("Project3_Dataset_v1.txt", sep=" ", names=["x", "y", "z"])
fit1 =  np.sort(np.array(df["z"]))
fit2 =  np.sort(np.array(df_given["z"]))
print(stats.ttest_ind(fit1, fit2))
print(fit1, fit2)
plt.plot(fit1, fit2)
plt.xlabel("Given dataset")
plt.ylabel("Generated dataset")
plt.title("QQ plot for Z value")
plt.show()

