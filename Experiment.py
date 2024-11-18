#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
import time
from Reinforce import Reinforce_Main
from ActorCritic import ActorCritic 
from Helper import LearningCurvePlot, smooth

def average_over_repetitions_Reinforce(n_repetitions, num_timesteps, learning_rate, gamma, entropy, smoothing_window):
    returns_over_repetitions = []
    now = time.time()
    for repetition in range(n_repetitions):
        print("Repetition Number: ", repetition + 1)
        returns, timesteps = Reinforce_Main(
            num_timesteps = num_timesteps,  
            learning_rate=learning_rate,
            gamma=gamma,
            entropy = entropy
        )
        returns_over_repetitions.append(returns)
    
    print('Running one setting takes {} minutes'.format((time.time()-now)/60))
    learning_curve = np.mean(np.array(returns_over_repetitions),axis=0) # average over repetitions  
    if smoothing_window is not None: 
        learning_curve = smooth(learning_curve,smoothing_window) # additional smoothing
    return learning_curve, timesteps  

def average_over_repetitions_ActorCritic(n_repetitions, num_timesteps, method, learning_rate, gamma, entropy, n_steps, smoothing_window):
    returns_over_repetitions = []
    now = time.time()
    for repetition in range(n_repetitions):
        print("Repetition Number: ", repetition + 1)
        returns, timesteps = ActorCritic(
            num_timesteps = num_timesteps, 
            method = method,
            learning_rate = learning_rate,
            gamma = gamma,
            entropy = entropy,
            n_steps = n_steps
        )
        returns_over_repetitions.append(returns)
    
    print('Running one setting takes {} minutes'.format((time.time()-now)/60))
    learning_curve = np.mean(np.array(returns_over_repetitions),axis=0) # average over repetitions  
    if smoothing_window is not None: 
        learning_curve = smooth(learning_curve,smoothing_window) # additional smoothing
    return learning_curve, timesteps  



def experiment():

    n_repetitions = 1
    num_timesteps = 1000000
    learning_rate = 0.001
    gamma = 0.99
    entropy = 0.1
    smoothing_window = 11
    
    learning_rates = [0.001, 0.0005, 0.0001]
    Plot = LearningCurvePlot(title = 'REINFORCE: Learning Rate') 
    Plot.set_ylim(-150,150) 
    for learning_rate in learning_rates:
        avg_learning_curve, timesteps = average_over_repetitions_Reinforce(n_repetitions, num_timesteps, learning_rate, gamma, entropy, smoothing_window)
        Plot.add_curve(timesteps, avg_learning_curve,label=r' Learning Rate = {}'.format(learning_rate))
    Plot.save('LearningRateVariation.png')
    
    learning_rate = 0.001
    entropys = [0.1, 0.01, 0.001]
    Plot = LearningCurvePlot(title = 'REINFORCE: Entropy Regularization Strength') 
    Plot.set_ylim(-150,100) 
    for entropy in entropys:
        avg_learning_curve, timesteps = average_over_repetitions_Reinforce(n_repetitions, num_timesteps, learning_rate, gamma, entropy, smoothing_window)
        Plot.add_curve(timesteps, avg_learning_curve,label=r' Entropy = {}'.format(entropy))
    Plot.save('EntropyVariation.png')
    
    learning_rate = 0.001
    entropy = 0.1
    gammas = [0.99, 0.95, 0.9]
    Plot = LearningCurvePlot(title = 'REINFORCE: Discount Factor') 
    Plot.set_ylim(-200,100) 
    for gamma in gammas:
        avg_learning_curve, timesteps= average_over_repetitions_Reinforce(n_repetitions, num_timesteps, learning_rate, gamma, entropy, smoothing_window)
        Plot.add_curve(timesteps, avg_learning_curve,label=r' Discount Factor = {}'.format(gamma))
    Plot.save('DiscountFactor.png')
    
    num_timesteps = 3000000
    learning_rate = 0.001
    entropy = 0.1
    gamma = 0.99
    smoothing_window = 11
    Plot = LearningCurvePlot(title = 'REINFORCE on Lunar Lander') 
    Plot.set_ylim(-100,200) 
    avg_learning_curve, timesteps = average_over_repetitions_Reinforce(n_repetitions, num_timesteps, learning_rate, gamma, entropy, smoothing_window)
    Plot.add_curve(timesteps, avg_learning_curve)
    Plot.save('REINFORCE.png')
    
    
    n_repetitions = 1
    num_timesteps = 1000000
    method = "Bootstrapping + baseline subtraction"
    learning_rate = 0.001
    gamma = 0.95
    entropy = 0.1
    n_steps = 5
    smoothing_window = 11
    
    learning_rates = [ 0.001, 0.0005,  0.0001]
    n_steps = 5
    Plot = LearningCurvePlot(title = 'Actor-Critic: Learning Rate') 
    Plot.set_ylim(-200,100) 
    for learning_rate in learning_rates:
        avg_learning_curve, timesteps = average_over_repetitions_ActorCritic(n_repetitions, num_timesteps, method, learning_rate, gamma, entropy, n_steps, smoothing_window)
        Plot.add_curve(timesteps, avg_learning_curve,label=r' Learning Rate = {}'.format(learning_rate))
    Plot.save('LearningRateVariation_Actor-Critic.png')
    
    entropys = [0.1, 0.01, 0.001]
    n_steps = 5
    Plot = LearningCurvePlot(title = 'Actor-Critic: Entropy Regularization Strength') 
    Plot.set_ylim(-500,100) 
    for entropy in entropys:
        avg_learning_curve, timesteps = average_over_repetitions_ActorCritic(n_repetitions, num_timesteps, method, learning_rate, gamma, entropy, n_steps, smoothing_window)
        Plot.add_curve(timesteps, avg_learning_curve,label=r' Entropy = {}'.format(entropy))
    Plot.save('EntropyVariation_Actor-Critic.png')
    
    gammas = [0.99, 0.95, 0.9]
    n_steps = 5
    Plot = LearningCurvePlot(title = 'Actor-Critic: Entropy Regularization Strength') 
    Plot.set_ylim(-300,50) 
    for gamma in gammas:
        avg_learning_curve, timesteps = average_over_repetitions_ActorCritic(n_repetitions, num_timesteps, method, learning_rate, gamma, entropy, n_steps, smoothing_window)
        Plot.add_curve(timesteps, avg_learning_curve,label=r' Discount Factor = {}'.format(gamma))
    Plot.save('DiscountFactor_Actor-Critic.png')
    
    n_steps = [1, 5, 10]
    Plot = LearningCurvePlot(title = 'Actor-Critic: Entropy Regularization Strength') 
    Plot.set_ylim(-300,50) 
    for n_step in n_steps:
        avg_learning_curve, timesteps = average_over_repetitions_ActorCritic(n_repetitions, num_timesteps, method, learning_rate, gamma, entropy, n_step, smoothing_window)
        Plot.add_curve(timesteps, avg_learning_curve,label=r'{}-step'.format(n_step))
    Plot.save('Nstep_Actor-Critic.png')
    
    num_timesteps = 1500000
    learning_rate = 0.001
    gamma = 0.95
    entropy = 0.1
    smoothing_window = 11
    n_steps = 5
    methods = ["Bootstrapping", "Baseline subtraction", "Bootstrapping + baseline subtraction"]
    
    Plot = LearningCurvePlot(title = 'Actor-Critic Variants on Lunar Lander') 
    Plot.set_ylim(-200,150) 
    for method in methods:
        avg_learning_curve, timesteps = average_over_repetitions_ActorCritic(n_repetitions, num_timesteps, method, learning_rate, gamma, entropy, n_steps, smoothing_window)
        Plot.add_curve(timesteps, avg_learning_curve,label=r' Actor Critic  = {}'.format(method))
    Plot.save('Actor-Critic.png')
    
    
    num_timesteps = 1000000
    learning_rate = 0.001
    gamma = 0.99
    entropy = 0.1
    smoothing_window = 11
    n_steps = 5
    method = "Bootstrapping + baseline subtraction"
    Plot = LearningCurvePlot(title = 'Lunar Lander Algorithm variation') 
    Plot.set_ylim(-200,150) 
    avg_learning_curve, timesteps = average_over_repetitions_Reinforce(n_repetitions, num_timesteps, learning_rate, gamma, entropy, smoothing_window)
    Plot.add_curve(timesteps, avg_learning_curve,label=r'REINFORCE')
    
    learning_rate = 0.001
    gamma = 0.99
    entropy = 0.1
    n_steps = 5
    method = "Bootstrapping + baseline subtraction"
    avg_learning_curve, timesteps = average_over_repetitions_ActorCritic(n_repetitions, num_timesteps, method, learning_rate, gamma, entropy, n_steps, smoothing_window)
    Plot.add_curve(timesteps, avg_learning_curve,label=r'Actor-Critic')
    
    Plot.save('Variation.png')
    

if __name__ == '__main__':
    experiment()  