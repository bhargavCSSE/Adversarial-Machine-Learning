#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 16:17:15 2019

@author: arminkhayyer, yewang, bhargavjoshi
"""

import os
import random
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from operator import attrgetter

class AParticle:
    def __init__(self, dimension):
        self.dimension = dimension
        self.position = []
        self.best_position = []
        self.velocity = []
        self.current_fitness = 0
        self.best_fitness = 0

    def randomly_generate(self, lb, ub, v_min, v_max):
        for i in range(self.dimension):
            self.position.append(random.uniform(lb, ub))
            self.velocity.append(random.uniform(v_min, v_max))
        self.current_fitness = 0

    def calculate_fitness(self):
        x2y2 = self.position[0] ** 2 + self.position[1] ** 2
        self.current_fitness = 0.5 + (math.sin(math.sqrt(x2y2)) ** 2 - 0.5) / (1 + 0.001 * x2y2) ** 2


    def calculate_velocity(self, best_particle, cognitive_rate, social_rate):
        phi = cognitive_rate + social_rate
        k = 2/(np.abs(2-phi - np.sqrt(phi**2 -4*phi)))
        for i in range(self.dimension):
            self.velocity[i] = k *(self.velocity[i] + (cognitive_rate * random.random()* (self.best_position[i] - self.position[i]))+
                                   (social_rate * random.random() * (best_particle.position[i] - self.position[i])))

    def find_best_position(self):
        if self.current_fitness >= self.best_fitness:
            self.best_position = [self.position[0], self.position[1]]
            self.best_fitness = self.current_fitness


    def calculate_new_position(self):
        for i in range(self.dimension):
            self.position[i] = self.position[i] + self.velocity[i]
            if self.position[i] >100:
                self.position[i] = 100
            if self.position[i] < -100:
                self.position[i] = -100

    def print_particle(self, i):
        print("current position " + str(i) + ": " + str(self.position) + " Fitness: " + str(self.current_fitness) + "    best_fit  " + str(self.best_fitness) + str(self.best_position))


class PSO:
    def __init__(self, swarm_size, dimension, lb, ub,v_max , v_min, cognitive_rate, social_rate, topology, sync_update):
        if (swarm_size < 2):
            print("Error: swarm Size must be greater than 2")
            sys.exit()
        self.swarm_size = swarm_size
        self.dimension = dimension
        self.lb = lb
        self.ub = ub
        self.Swarm = []
        self.hacker_tracker_x = []
        self.hacker_tracker_y = []
        self.hacker_tracker_z = []
        self.topology = topology
        self.sync_update = sync_update
        self.cognitive_rate = cognitive_rate
        self.social_rate = social_rate
        self.v_max = v_max
        self.v_min = v_min

    def generate_initial_population(self):
        for i in range(self.swarm_size):
            particle = AParticle(self.dimension)
            particle.randomly_generate(self.lb, self.ub, self.v_max, self.v_min)
            particle.calculate_fitness()
            particle.find_best_position()
            particle.print_particle(i)
            self.hacker_tracker_x.append(particle.position[0])
            self.hacker_tracker_y.append(particle.position[1])
            self.hacker_tracker_z.append(particle.current_fitness)
            self.Swarm.append(particle)

    def get_worst_fit_individual(self):
        worst_fitness = 999999999.0  # For Maximization
        worst_individual = -1
        for i in range(self.swarm_size):
            if (self.Swarm[i].current_fitness < worst_fitness):
                worst_fitness = self.Swarm[i].current_fitness
                worst_individual = i
        return worst_individual

    def get_best_fitness(self):
        best_fitness = -99999999999.0
        best_individual = -1
        for i in range(self.swarm_size):
            if self.Swarm[i].current_fitness > best_fitness:
                best_fitness = self.Swarm[i].current_fitness
                best_individual = i
        return best_fitness

    def find_best_neghibour(self, iterator):
        best_swarm = 0
        if self.topology == "ring":
            neighbours = [self.Swarm[iterator -1 ], self.Swarm[iterator], self.Swarm[(iterator+1)%self.swarm_size ]]
            best_swarm = max(neighbours, key=attrgetter("current_fitness"))
        if self.topology == "star":
            best_swarm = max(self.Swarm, key=attrgetter("current_fitness"))
        return best_swarm


    def Swarm_cycle(self):
        if self.sync_update == "synchronous":
            for i in range(len(self.Swarm)):
                best_neighbour = self.find_best_neghibour(i)
                self.Swarm[i].calculate_velocity(best_neighbour, self.cognitive_rate, self.social_rate)

            for particle in self.Swarm:
                particle.calculate_new_position()
                particle.calculate_fitness()
                particle.find_best_position()
                self.hacker_tracker_x.append(particle.position[0])
                self.hacker_tracker_y.append(particle.position[1])
                self.hacker_tracker_z.append(particle.current_fitness)

        if self.sync_update == "asynchronous":
            for i, particle in enumerate(self.Swarm):
                best_neighbour = self.find_best_neghibour(i)
                self.Swarm[i].calculate_velocity(best_neighbour, self.cognitive_rate, self.social_rate)
                particle.calculate_new_position()
                particle.calculate_fitness()
                particle.find_best_position()
                self.hacker_tracker_x.append(particle.position[0])
                self.hacker_tracker_y.append(particle.position[1])
                self.hacker_tracker_z.append(particle.current_fitness)

    def print_population(self):
        for i in range(self.swarm_size):
            self.Swarm[i].print_particle(i)

    def print_best_max_fitness(self):
        best_fitness = -999999999.0  # For Maximization
        best_individual = -1
        for i in range(self.swarm_size):
            if self.Swarm[i].best_fitness > best_fitness:
                best_fitness = self.Swarm[i].best_fitness
                best_individual = i
        print("Best Indvidual: ", str(best_individual), " ", self.Swarm[best_individual].best_position, " Fitness: ",
              str(best_fitness))

    def plot_evolved_candidate_solutions(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1, projection='3d')
        ax1.scatter(self.hacker_tracker_x, self.hacker_tracker_y, self.hacker_tracker_z)
        plt.title("Evolved Candidate Solutions")
        ax1.set_xlim3d(-100.0, 100.0)
        ax1.set_ylim3d(-100.0, 100.0)
        ax1.set_zlim3d(0.2, 1.0)
        plt.show()

class GUI_operation:
    def __init__(self):
        self.topology = 'null'
        self.update_method = 'null'
        self.algorithm_index = '0'
    
    def print_list(self):
        print("The list of PSO algorithms:")
        print("1. PSO, Ring Topology (Neighborhood Size = 3), Synchronous Update")
        print("2. PSO, Ring Topology (Neighborhood Size = 3), Asynchronous Update")
        print("3. PSO, Star Topology, Synchronous Update")
        print("4. PSO, Star Topology, Asynchronous Update")
    
    def algorithm_select(self):
        self.algorithm_index = input("Enter algorithm index from the list: ")
        
        if(self.algorithm_index == '1'):
            self.topology = "ring"
            self.update_method = "synchronous"
        elif(self.algorithm_index == '2'):
            self.topology = "ring"
            self.update_method = "asynchronous"
        elif(self.algorithm_index == '3'):
            self.topology = "star"
            self.update_method = "synchronous"
        elif(self.algorithm_index == '4'):
            self.topology = "star"
            self.update_method = "asynchronous"
        
        return self.topology, self.update_method
    
dimension = 2
ub = 100.0
lb = -100.0
MaxEvaluations = 4000
plot = 0
swarm_size = 100
total_runs = 1              #Change the number of runs for more than one consecutive runs
fail = 0

GUI_operation_begin = GUI_operation()
GUI_operation_begin.print_list()
topology, sync_method = GUI_operation_begin.algorithm_select()


df = pd.DataFrame(index=[i for i in range(total_runs)], columns=["Run", "Function_Evaluations"])
runs = 0
while runs < total_runs:
    simple_PSO = PSO(swarm_size, dimension, lb, ub,v_max=0,v_min=0, cognitive_rate=2.05, social_rate= 2.05, topology= topology, sync_update= sync_method )
    
    simple_PSO.generate_initial_population()
    for i in range(MaxEvaluations - swarm_size + 1):
        simple_PSO.Swarm_cycle()
        if (i % swarm_size == 0):
            if (plot == 1):
                simple_PSO.plot_evolved_candidate_solutions()
            print("At Iteration: " + str(i))
            simple_PSO.print_population()
        if (simple_PSO.get_best_fitness() >= 0.99754):
            break
    
    print("\nFinal Population\n")
    simple_PSO.print_population()
    simple_PSO.print_best_max_fitness()
    print("Function Evaluations: " + str(swarm_size + i))
    simple_PSO.plot_evolved_candidate_solutions()                       #Advised to comment this line if running code multiple times
    if (swarm_size + i) < 4000:
        df.iloc[runs]["Run"] = runs + 1
        df.iloc[runs]["Function_Evaluations"] = swarm_size + i
        runs = runs + 1
    else:
        fail = fail + 1
print("\nThe final data frame\n")   
print(df)
print("Totals fails = "+ str(fail))
df.to_csv(topology+'_'+sync_method+'.csv', sep=",")