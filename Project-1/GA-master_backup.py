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
    def __init__(self, population_size, chromosome_length, mutation_rate, lb, ub, crossover, strategy, ):
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
        self.strategy = strategy
        self.crossover = crossover

    def generate_initial_population(self):
        for i in range(self.population_size):
            individual = anIndividual(self.chromosome_length)
            individual.randomly_generate(self.lb,self.ub)
            individual.calculate_fitness()
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
    
    def get_best_fit_individual(self):
        best_fitness = -99999999999.0
        best_individual = -1
        for i in range(self.population_size):
            if self.population[i].fitness > best_fitness:
                best_fitness = self.population[i].fitness
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
        if self.crossover == "SPX":
            single_point = random.randint(1, self.chromosome_length-1)
            for j in range(self.chromosome_length):
                if j <= single_point:
                    kid.chromosome[j] = mom.chromosome[j]
                else: kid.chromosome[j] = dad.chromosome[j]

                kid.chromosome[j] += self.mutation_amt * random.gauss(0,1.0)
                if kid.chromosome[j] > self.ub:
                    kid.chromosome[j] = self.ub
                if kid.chromosome[j] < self.lb:
                    kid.chromosome[j] = self.lb

        elif self.crossover == "Midx":
            for j in range(self.chromosome_length):
                probabaility = random.uniform(0, 1)
                if probabaility <= 1:
                    kid.chromosome[j] = (mom.chromosome[j] + dad.chromosome[j])/2
                else:
                    kid.chromosome[j] = mom.chromosome[j]
                kid.chromosome[j] += self.mutation_amt * random.gauss(0, 1.0)
                if kid.chromosome[j] > self.ub:
                    kid.chromosome[j] = self.ub
                if kid.chromosome[j] < self.lb:
                    kid.chromosome[j] = self.lb

        elif self.crossover == "BLX_0.0":
            for j in range(self.chromosome_length):
                kid.chromosome[j] = random.uniform(mom.chromosome[j],dad.chromosome[j])
                kid.chromosome[j] += self.mutation_amt * random.gauss(0, 1.0)
                if kid.chromosome[j] > self.ub:
                    kid.chromosome[j] = self.ub
                if kid.chromosome[j] < self.lb:
                    kid.chromosome[j] = self.lb
        return kid

    def evolutionary_cycle(self):
        if self.strategy == "SteadyState":
            mom, dad  = self.tournoment_selection()
            worst_individual = self.get_worst_fit_individual()
            self.population.pop(worst_individual)
            kid = self.Crossover_operator(mom, dad)
            self.population.append(kid)
            kid.calculate_fitness()
            self.hacker_tracker_x.append(kid.chromosome[0])
            self.hacker_tracker_y.append(kid.chromosome[1])
            self.hacker_tracker_z.append(kid.fitness)

        elif self.strategy == "Mu+1":
            mom, dad  = self.tournoment_selection()
            kid = self.Crossover_operator(mom, dad)
            self.population.append(kid)
            kid.calculate_fitness()
            worst_individual = self.get_worst_fit_individual()
            if worst_individual != self.population.index(kid):
                self.hacker_tracker_x.append(kid.chromosome[0])
                self.hacker_tracker_y.append(kid.chromosome[1])
                self.hacker_tracker_z.append(kid.fitness)
            self.population.pop(worst_individual)
        
        elif self.strategy == "ElitistGen-GA":
            population_kids = []
            elite_individual = self.get_best_fit_individual()
            for i in range(self.population_size):
                mom, dad  = self.tournoment_selection()
                kid = self.Crossover_operator(mom, dad)
                population_kids.append(kid)
                kid.calculate_fitness()
                self.hacker_tracker_x.append(kid.chromosome[0])
                self.hacker_tracker_y.append(kid.chromosome[1])
                self.hacker_tracker_z.append(kid.fitness)
            self.population = population_kids
            worst_individual = self.get_worst_fit_individual()
            self.population.pop(worst_individual)
            self.population.append(elite_individual)
            
        elif self.strategy == "SteadyGen-GA_version_bhargav":
            intermediate = []
            kids = []
            for i in range(self.population_size):
                mom, dad = self.tournoment_selection()
                kid = self.Crossover_operator(mom, dad)
                kids.append(kid)
                kid.calculate_fitness()
                self.hacker_tracker_x.append(kid.chromosome[0])
                self.hacker_tracker_y.append(kid.chromosome[1])
                self.hacker_tracker_z.append(kid.fitness)
            worst_individual = self.get_worst_fit_individual()
            self.population.pop(worst_individual)
            intermediate = self.population
            self.population = kids
            best_fit_kid = self.get_best_fit_individual()
            self.population = intermediate
            self.population.append(best_fit_kid)


        elif self.strategy == "SteadyGen-GA_version_Armin":
            mom, dad = self.tournoment_selection()
            kid = self.Crossover_operator(mom, dad)
            kid.calculate_fitness()
            self.hacker_tracker_x.append(kid.chromosome[0])
            self.hacker_tracker_y.append(kid.chromosome[1])
            self.hacker_tracker_z.append(kid.fitness)
            elite_individual = self.get_best_fit_individual()
            self.population.remove(elite_individual)
            worst_individual_random = random.choice(self.population)
            self.population.remove(worst_individual_random)
            self.population.append(elite_individual)
            self.population.append(kid)


        elif self.strategy == "Mu+Mu":
            elite_individual = self.get_best_fit_individual()
            for i in range(self.population_size):
                mom, dad  = self.tournoment_selection()
                kid = self.Crossover_operator(mom, dad)
                self.population.append(kid)
                kid.calculate_fitness()
                self.hacker_tracker_x.append(kid.chromosome[0])
                self.hacker_tracker_y.append(kid.chromosome[1])
                self.hacker_tracker_z.append(kid.fitness)
            for i in range(self.population_size):
                worst_individual = self.get_worst_fit_individual()
                self.population.pop(worst_individual)
        else: raise ValueError('Make sure the strategy is selected correctly')

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
        return best_fitness
    
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
MaxEvaluations = 4000
plot = 0

PopSize = 15
mu_amt  = 0.01



df = pd.DataFrame(index=[i for i in range(50)], columns=[ "Run", "SPX_best", "SPX_Function_Evaluations","Midx_best", 	"Midx_Function_Evaluations", "BLX_0.0_best", "BLX_0.0_Function_Evaluations"])
strategy = "Mu+1"
for k in range(50):
    df.iloc[k]["Run"] = k
    for j in ["SPX", "Midx", "BLX_0.0" ]:
        simple_exploratory_attacker = aSimpleExploratoryAttacker(PopSize,ChromLength,mu_amt,lb,ub, crossover=j, strategy=strategy )

        simple_exploratory_attacker.generate_initial_population()
        simple_exploratory_attacker.print_population()

        for i in range(MaxEvaluations-PopSize+1):
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
        best = simple_exploratory_attacker.print_best_max_fitness()
        print("Function Evaluations: " + str(PopSize+i))
        simple_exploratory_attacker.plot_evolved_candidate_solutions()
        df.iloc[k][j+"_best"] = best
        df.iloc[k][j + "_Function_Evaluations"] = PopSize+i



print(df)
df.to_csv(strategy+'.csv', sep=",")