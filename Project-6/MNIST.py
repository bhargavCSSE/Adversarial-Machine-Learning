# -*- coding: utf-8 -*-
"""
@author: Bhargav Joshi

@Base code provided by Gerry Dozier
"""

import os
import random
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.utils import np_utils, plot_model
from matplotlib import pyplot as plt
from warnings import simplefilter

class anIndividual:
    def __init__(self, specified_chromosome_length):
        self.chromosome = []
        self.fitness    = 0
        self.chromosome_length = specified_chromosome_length
        
    def randomly_generate(self,lb, ub):
        for i in range(self.chromosome_length):
            self.chromosome.append(random.choice([0, random.randint(10, ub)]))
    
    def calculate_fitness(self):
        self.fitness = NeuralNetwork(self.chromosome)

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


    def tournament_selection(self, k=2):
        tournament_output1 = random.choices(self.population, k=k)

        best_individual1 = [i.fitness for i in tournament_output1]
        parent1 = tournament_output1[best_individual1.index(max(best_individual1))]

        tournament_output2 = random.choices(self.population, k=k)
        
        best_individual2 = [i.fitness for i in tournament_output2]
        parent2 = tournament_output2[best_individual2.index(max(best_individual2))]
       
        return parent1, parent2


    def Crossover_operator(self, mom, dad):
        kid = anIndividual(self.chromosome_length)
        kid.randomly_generate(self.lb, self.ub)
        if self.crossover == "Midx":
            for j in range(self.chromosome_length):
                probability = random.uniform(0, 1)
                if probability <= 1:
                    kid.chromosome[j] = int((mom.chromosome[j] + dad.chromosome[j])/2)
                else:
                    kid.chromosome[j] = mom.chromosome[j]
                kid.chromosome[j] += int(self.mutation_amt * random.gauss(0, 1.0))
                if kid.chromosome[j] > self.ub:
                    kid.chromosome[j] = self.ub
                if kid.chromosome[j] < self.lb:
                    kid.chromosome[j] = self.lb

        elif self.crossover == "BLX_0.0":
            for j in range(self.chromosome_length):
                kid.chromosome[j] = int(random.uniform(mom.chromosome[j],dad.chromosome[j]))
                kid.chromosome[j] += int(self.mutation_amt * random.gauss(0.0, 1.0))
                if kid.chromosome[j] > self.ub:
                    kid.chromosome[j] = self.ub
                if kid.chromosome[j] < self.lb:
                    kid.chromosome[j] = self.lb
        return kid

    def evolutionary_cycle(self):
        
        if self.strategy == "ElitistGen-GA":
            population_kids = []
            elite_individual = self.get_best_fit_individual()
            for i in range(self.population_size):
                mom, dad  = self.tournament_selection()
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
            
        elif self.strategy == "SteadyState":
            mom, dad  = self.tournament_selection()
            worst_individual = self.get_worst_fit_individual()
            self.population.pop(worst_individual)
            kid = self.Crossover_operator(mom, dad)
            self.population.append(kid)
            kid.calculate_fitness()
            self.hacker_tracker_x.append(kid.chromosome[0])
            self.hacker_tracker_y.append(kid.chromosome[1])
            self.hacker_tracker_z.append(kid.fitness)

        elif self.strategy == "Mu+1":
            mom, dad  = self.tournament_selection()
            kid = self.Crossover_operator(mom, dad)
            self.population.append(kid)
            kid.calculate_fitness()
            worst_individual = self.get_worst_fit_individual()
            if worst_individual != self.population.index(kid):
                self.hacker_tracker_x.append(kid.chromosome[0])
                self.hacker_tracker_y.append(kid.chromosome[1])
                self.hacker_tracker_z.append(kid.fitness)
            self.population.pop(worst_individual)
            
        elif self.strategy == "SteadyGen-GA_version_Bhargav":
            intermediate = []
            kids = []
            for i in range(self.population_size):
                mom, dad = self.tournament_selection()
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
            mom, dad = self.tournament_selection()
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
                mom, dad  = self.tournament_selection()
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
        return best_fitness, self.population[best_individual].chromosome
    
    def plot_evolved_candidate_solutions(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1,projection='3d')
        ax1.scatter(self.hacker_tracker_x,self.hacker_tracker_y,self.hacker_tracker_z)
        plt.title("Evolved Candidate Solutions")
        ax1.set_xlim3d(-100.0,100.0)
        ax1.set_ylim3d(-100.0,100.0)
        ax1.set_zlim3d(0.2,1.0)
        plt.show()

def crossover_select(crossover_selector):
    if crossover_selector == '1':
        crossover = "SPX"
        return crossover
    elif crossover_selector == '2':
        crossover = "Midx"
        return crossover
    elif crossover_selector == '3':
        crossover = "BLX_0.0"
        return crossover
        
def strategy_select(strategy_selector):
    if strategy_selector == '1':
        strategy = "ElitistGen-GA"
        return strategy
    elif strategy_selector == '2':
        strategy = "SteadyState"
        return strategy
    elif strategy_selector == '3':
        strategy = "SteadyGen-GA_version_Armin"
        return strategy
    elif strategy_selector == '4':
        strategy = "Mu+1"
        return strategy
    elif strategy_selector == '5':
        strategy = "Mu+Mu"
        return strategy

def NeuralNetwork(LayerList):

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
    LayerCount = 0
    print("LayerList\n" + str(LayerList))
    NeuralNetwork = tf.keras.Sequential()
    NeuralNetwork.add(tf.keras.layers.Flatten(name='InputLayer'))
    for i in range(len(LayerList)):
        if int(LayerList[i]) and int(LayerList[i]) > 10:
            NeuralNetwork.add(tf.keras.layers.Dense(int(LayerList[i]), activation=tf.nn.relu))
            LayerCount += 1
    NeuralNetwork.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax, name='OutputLayer'))
    print("Layers Added: "+ str(LayerCount))

    # Train Model
    NeuralNetwork.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    NeuralNetwork.fit(X_train, y_train, epochs=1, use_multiprocessing=True)
    plot_model(NeuralNetwork, to_file='NeuralNetwork.png')

    # Test Model
    y_pred = NeuralNetwork.predict(X_train)
    val_loss, val_acc = NeuralNetwork.evaluate(X_test, y_test)
    print("Test loss:" + str(val_loss) + "   " + "Test acc:" + str(val_acc))
    print(np.argmax(y_pred[0]))
    return(val_acc)


ChromLength = 5
ub = 150
lb = 0
MaxEvaluations = 30
PopSize = 5
mu_amt = 0.05
runCount = 1

HiddenLayer = []
df = pd.DataFrame(columns=["Run", "HiddenLayer", "Accuracy"])
for run in range(runCount):
    simple_exploratory_attacker = aSimpleExploratoryAttacker(PopSize, ChromLength, mu_amt, lb, ub,
                                                             crossover="BLX_0.0",
                                                             strategy="Mu+1", )

    simple_exploratory_attacker.generate_initial_population()
    simple_exploratory_attacker.print_population()

    for i in range(MaxEvaluations):
        simple_exploratory_attacker.evolutionary_cycle()
        if (i % PopSize == 0):
            print("At Iteration: " + str(i))
            simple_exploratory_attacker.print_population()
        if (simple_exploratory_attacker.get_best_fitness() >= 0.99754):
            break

    print("\nFinal Population\n")
    simple_exploratory_attacker.print_population()
    fit, chromosome = simple_exploratory_attacker.print_best_max_fitness()
    print("Function Evaluations: " + str(PopSize + i))

    for j in range(len(chromosome)):
        if chromosome[j] and chromosome[j]>10:
            HiddenLayer.append(chromosome[j])

    df = df.append(dict(Run=run+1,
                        HiddenLayer=HiddenLayer,
                        Accuracy=fit), ignore_index=True)
    HiddenLayer = []

print(df)
df.to_csv('Results_MNIST.csv')