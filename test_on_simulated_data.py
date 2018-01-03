import matplotlib
from constants_and_util import *
import numpy as np
import random
from IPython import embed
import time
import pickle
import cyclic_HMM
from scipy.stats import pearsonr, spearmanr
from multiprocessing import Process, Manager
import inspect
import os
import matplotlib.pyplot as plt 

"""
Methods for generating simulated data. 
First define a few helper methods. 
"""

def sigmoid(x): 
    # checked. 
    return 1 / (1 + np.exp(-x))
def logit(p): # checked. 
    if p == 0:
        return -np.inf
    if p == 1:
        return np.inf
    return np.log(p) - np.log(1 - p)
def random_centered_matrix(shape):
    return np.random.random(shape) - .5

class simulatedDataGenerator():
    """
    This class generates simulated data. 
    """
    def __init__(self, param_setting):
        self.__dict__ = param_setting.__dict__.copy()
        self.symptoms = ['symptom_%i' % k for k in range(self.K)]

    def generate_population_params(self): 
        # checked. Generates overall sinusoid coefficients for the population. 
        self.population_sinusoid_coefs = random_centered_matrix([self.K, self.n_sinusoids]) * 10
        self.population_sinusoid_phases = random_centered_matrix([self.K, self.n_sinusoids]) * 2 * np.pi
        if self.data_is_binary:
            self.population_missing_data_sinusoid_coefs = random_centered_matrix([1, self.n_sinusoids])
        else:
            self.population_missing_data_sinusoid_coefs = random_centered_matrix([self.K, self.n_sinusoids])

    def generate_individual_params(self):
        # checked. Generates individual cycle parameters -- coefficients and start day and cycle length. 
        individual_sinusoid_coefs =  self.population_sinusoid_coefs + random_centered_matrix(self.population_sinusoid_coefs.shape) * self.sinusoid_coefficient_variation 
        individual_missing_data_sinusoid_coefs =  self.population_missing_data_sinusoid_coefs + random_centered_matrix(self.population_missing_data_sinusoid_coefs.shape) * self.sinusoid_coefficient_variation 

        T = random.choice(range(self.min_T, self.max_T + 1))
        cycle_length = max(5, int(np.random.normal(loc = self.population_cycle_length, scale = self.cycle_variance_between_individuals))) # keep cycle length from being too 
        start_day = random.choice(range(cycle_length))
        d = {'missing_data_sinusoid_coefs':individual_missing_data_sinusoid_coefs, 'sinusoid_coefs':individual_sinusoid_coefs, 'T':T, 'cycle_length':cycle_length, 'start_day':start_day}
        return d

    def generate_individual_data(self, individual_params):
        # Checked. Draws an individual sample given the individual generative parameters. 
        # First, generate phases. 
        current_cycle_length = max(individual_params['start_day'] + 1, int(np.random.normal(individual_params['cycle_length'], self.cycle_variance_within_individuals)))
        cycle_days = []
        phases = []
        current_cycle_day = individual_params['start_day']
        while len(cycle_days) < individual_params['T']:
            current_cycle_length = max(current_cycle_length, 5) # make sure at least 5 -- some simulation params are weird. 
            cycle_days.append(current_cycle_day)
            phases.append(2.*np.pi*current_cycle_day / current_cycle_length)
            current_cycle_day += 1
            if current_cycle_day >= current_cycle_length:
                current_cycle_day = 0
                current_cycle_length = int(np.random.normal(individual_params['cycle_length'], self.cycle_variance_within_individuals))
        phases = np.array(phases)
        cycle_days = np.array(cycle_days)

        # Second, draw samples given phases.
        sample = self.compute_sinusoid_values(individual_params['sinusoid_coefs'], phases, self.noise)
        if not self.data_is_binary:
            sample = sample + 5 # add offset so it isn't centered at 0 (because this messes up variability calculation)
        missing_data_probability = sigmoid(self.compute_sinusoid_values(individual_params['missing_data_sinusoid_coefs'], phases, noise = 0) + logit(self.missing_data_probability))
        assert(len(phases) == individual_params['T'])
        assert(sample.shape[0] == individual_params['T'])
        assert(sample.shape[1] == self.K)

        # Third, add in missing data. 
        if self.data_is_binary:
            p = sigmoid(sample)
            sample = (np.random.random(sample.shape) < p) * 1.
            data_is_missing = np.random.random(missing_data_probability.shape) < missing_data_probability
            assert(data_is_missing.shape[0] == individual_params['T'])
            assert(data_is_missing.shape[1] == 1)
            for t in range(len(data_is_missing)):
                if data_is_missing[t]:
                    sample[t, :] = missing_data_val
            return {'phases':phases, 'sample':sample, 'cycle_days':cycle_days, 'p':p, 'missing_data_p':missing_data_probability}
        else:
            assert(missing_data_probability.shape == sample.shape)
            sample[np.random.random(missing_data_probability.shape) < missing_data_probability] = missing_data_val
            return {'phases':phases, 'sample':sample, 'cycle_days':cycle_days, 'missing_data_p':missing_data_probability}

      
    def compute_sinusoid_values(self, sinusoid_coefs, phases, noise):
        # Checked. 
        # sinusoid_coefs is number_of_output_dimensions x number_of_sinusoids. 
        n_output_dimensions, n_sinusoids = sinusoid_coefs.shape
        sample = np.zeros([len(phases), n_output_dimensions])
        for k in range(n_output_dimensions): # number of output dimensions. 
            for sinusoid in range(n_sinusoids): # number of sinusoids. 
                sample[:, k] += sinusoid_coefs[k, sinusoid] * np.sin(phases * (sinusoid + 1) + self.population_sinusoid_phases[k, sinusoid])
        sample = np.array(sample)
        sample = sample + noise * random_centered_matrix(sample.shape)
        return sample

    def plot_random_signal(self):
        # Plots the signal. Checked. 
        sample = random.choice(self.all_individual_data)['sample']
        plt.figure(figsize = [10, 10])
        for k in range(min(10, sample.shape[1])):
            plt.subplot(5, 2, k + 1)
            good_idxs = sample[:, k] != missing_data_val
            x = sample[:, k].copy()
            x[~good_idxs] = np.nan
            plt.plot(x)
            if self.data_is_binary:
                plt.ylim([-.05, 1.05])
        plt.suptitle("Simulated data for a random individual\nEach plot is one feature; x-axis is timestep\nMissing values in the data appear as gaps in the plot")

    def generate_simulated_data(self):
        # generates the simulated data. Checked. 
        self.generate_population_params()
        self.all_individual_params = []
        self.all_individual_data = []
        for i in range(self.N):
            individual_params = self.generate_individual_params()
            self.all_individual_params.append(individual_params)
            self.all_individual_data.append(self.generate_individual_data(individual_params))
        self.plot_random_signal()

class paramSetting():
    def __init__(self, d):
        self.__dict__ = d.copy()
    def __eq__(self, other):
        # equality comparison: two are the same if they have everything equal except for the name (which is just used as an experiment label). 
        d1 = self.__dict__.copy()
        d2 = other.__dict__.copy()
        del d1['name']
        del d2['name']
        return d1 == d2

def generate_param_settings(N = 500, K = 10, cycle_variance_between_individuals = 2, cycle_variance_within_individuals = 2, noise = 25, missing_data_probability = .2, population_cycle_length = 30, name = None, min_T = 90, max_T = 120, n_sinusoids = 1, sinusoid_coefficient_variation = .2, verbose = True):
    # Checked. Generates a simulation config, using the default kwargs above. 
    n_trials_per_iteration = 5
    all_param_settings = []
    for data_is_binary in [False, True]:
        for trial_number in range(n_trials_per_iteration):
            all_param_settings.append(paramSetting({'N':N, 'K':K, 'noise':noise, 'missing_data_probability': missing_data_probability, 'cycle_variance_between_individuals':cycle_variance_between_individuals, 'cycle_variance_within_individuals': cycle_variance_within_individuals, 'population_cycle_length':population_cycle_length, 'name':name, 'data_is_binary':data_is_binary, 'trial_number':trial_number, 'min_T':min_T, 'max_T':max_T, 'n_sinusoids':n_sinusoids, 'verbose':verbose, 'sinusoid_coefficient_variation':sinusoid_coefficient_variation}))
    return all_param_settings
    

def generate_params_for_all_tests():
    # Generate the parameters for the experiments we want to run. 
    
    all_param_settings = []

    # Vary max timeseries length. 
    for max_T in [90, 120, 150, 180]:
        all_param_settings += generate_param_settings(max_T = max_T, name = 'max_T')

    # Vary between-individual heterogeneity in cycle length. 
    for cycle_variance_between_individuals in [1, 2, 5, 10]:
        all_param_settings += generate_param_settings(cycle_variance_between_individuals = cycle_variance_between_individuals, name = 'cycle_variance_between_individuals')

    # Vary within-individual heterogeneity in cycle length. 
    for cycle_variance_within_individuals in [1, 2, 5, 10]:
        all_param_settings += generate_param_settings(cycle_variance_within_individuals = cycle_variance_within_individuals, name = 'cycle_variance_within_individuals')
    
    # Vary missing data. 
    for missing_data_probability in [0, .05, .1, .2, .4, .6, .8, .9]:
        all_param_settings += generate_param_settings(missing_data_probability = missing_data_probability, name = 'missing_data_probability')
    
    # Vary noise. 
    for noise in [5, 10, 25, 50]:
        all_param_settings += generate_param_settings(noise = noise, name = 'noise')

    return all_param_settings

    


