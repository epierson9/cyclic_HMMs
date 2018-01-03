import os
import numpy as np
import json, pickle
import numpy as np
import copy
import random
from IPython import embed

# because the HMM library we rely on, Pomegranate, does not deal well with nans or strings
# we encode missing data as an integer that will not appear in real data. This constant is used throughout the code. 
missing_data_val = -9999999 

def get_symptom_variability(vector):
    # Given a vector of symptom values, returns a measure of how variable the symptom is. 
    return np.mean(np.abs(vector - vector.mean()) / vector.mean())

def get_true_symptom_values_by_cycle_date(dates_relative_to_cycle_start, symptom_vectors):
    # Checked. 
    # arguments are a list n_people long of dates relative to cycle start
    # and a list n_people long of symptom vectors on those dates. 
    relative_dates_to_symptom_values = {}
    for i in range(len(dates_relative_to_cycle_start)):
        for t in range(len(dates_relative_to_cycle_start[i])):
            relative_date = dates_relative_to_cycle_start[i][t]
            if relative_date not in relative_dates_to_symptom_values:
                relative_dates_to_symptom_values[relative_date] = []
            relative_dates_to_symptom_values[relative_date].append(symptom_vectors[i][t])
    return relative_dates_to_symptom_values

def get_predicted_symptom_trajectories(start_state_name, model, n_timesteps):
    # start state is the state to project forward from
    # model is the model to use
    # n_timesteps is how many timesteps to project forward. 
    # sample usage: get_predicted_symptom_trajectories('state_0_duration_0', model, n_timesteps = n_timesteps_in_cycle)
    # Checked. 
    emission_distribution_name = model.emission_distribution_name
    assert(emission_distribution_name in ['bernoulli_with_missing_data', 'normal_with_missing_data'])
    good_state_idxs = []
    for idx in range(len(model.hmm.states)):
        state_name = model.hmm.states[idx].name
        if not ('start' in state_name or 'end' in state_name):
            good_state_idxs.append(idx)
        else:
            assert(idx >= len(model.hmm.states) - 2)
            
    # create emission probability matrix
    if emission_distribution_name == 'bernoulli_with_missing_data':
        # This will be n_substates x n_symptoms + 1, where the last row is missing data probability. 
        states_to_emission_probs = {}
        for j in range(model.n_states):
            emission_distribution = model.emission_distributions[j]
            states_to_emission_probs[j] = list(emission_distribution.emission_parameters['p'])
            states_to_emission_probs[j] = states_to_emission_probs[j] + [emission_distribution.emission_parameters['p_missing']]
            states_to_emission_probs[j] = np.array(states_to_emission_probs[j])
        emission_matrix = []
        for idx in range(len(model.hmm.states)):
            state_name = model.hmm.states[idx].name
            if not ('start' in state_name or 'end' in state_name):
                j = int(state_name.split('_')[1])
                emission_matrix.append(states_to_emission_probs[j])
        emission_matrix = np.array(emission_matrix)
    else:
        states_to_emission_params = {}
        for idx in range(len(model.hmm.states)):
            state_name = model.hmm.states[idx].name
            if not ('start' in state_name or 'end' in state_name):
                j = int(state_name.split('_')[1])
                states_to_emission_params[idx] = {'mu':copy.deepcopy(model.emission_distributions[j].emission_parameters['mu']), 
                                                  'sigma':copy.deepcopy(model.emission_distributions[j].emission_parameters['sigma']), 
                                                  'p_missing':copy.deepcopy(model.emission_distributions[j].emission_parameters['p_missing'])}
    # now compute predicted states by propagating forward. 
    # predicted_states is n_timesteps x n_substates (a probability distribution at each timestep)
    state_00_idx = [model.hmm.states[idx].name for idx in good_state_idxs].index(start_state_name)
    start_state_vector = np.zeros(len(good_state_idxs),)
    start_state_vector[state_00_idx] = 1.
    T = copy.deepcopy(model.hmm.dense_transition_matrix().transpose())
    T = T[good_state_idxs, :]
    T = T[:, good_state_idxs]
    predicted_states = [start_state_vector]
    
    for t in range(0, n_timesteps):
        next_state = np.dot(T, predicted_states[-1])
        predicted_states.append(next_state)
    predicted_states = np.array(predicted_states)
    
    if emission_distribution_name == 'bernoulli_with_missing_data':
        # dot together to get emission probabilities. 
        # predicted_symptom_values is n_symptoms + 1 x n_timesteps. 
        # (last row is missing data)
        predicted_symptom_values = np.dot(predicted_states, emission_matrix).transpose()
    else:
        predicted_symptom_values = []
        for k in range(len(model.symptom_names)):
            sampled_means_for_symptom = []
            for t in range(predicted_states.shape[0]):
                state_probs = predicted_states[t, :]
                samples = []
                for idx in range(len(state_probs)):
                    n_samples_to_draw = int(state_probs[idx] * 10000.)
                    mu = states_to_emission_params[idx]['mu'][k]
                    sigma = states_to_emission_params[idx]['sigma'][k]
                    samples = samples + list(np.random.normal(size = n_samples_to_draw, loc = mu, scale = sigma))
                sampled_means_for_symptom.append(np.mean(samples))
            predicted_symptom_values.append(sampled_means_for_symptom)
        predicted_symptom_values = np.array(predicted_symptom_values)
    # discard first timestep. 
    predicted_symptom_values = predicted_symptom_values[:, 1:]
    
    return predicted_symptom_values

def get_cycle_lengths(x): 
    # checked. Given a vector of ones and zeros, returns gaps between 1s. 
    return np.diff(np.flatnonzero(x == 1))

def extract_stage_starts(x, min_days_without_period = 3): # checked.
    # min_days_without_period denotes the smallest gap before we can say the stage has started again. 
    # this avoids very short cycles due to noisy data (eg, people logging periods and then missing a day)
    # this means that we will not extract a stage start if it occurs in the first min_days_without_period of the timeseries.
    # you may wish to change this for your data. 
    x = np.array(x)
    stage_starts = 0 * x
    period_hasnt_started = False
    days_without_period = 0
    for t in range(len(x)):
        if x[t] == 1 and period_hasnt_started:
            period_hasnt_started = False
            days_without_period = 0
            stage_starts[t] = 1
        if x[t] == 0:
            days_without_period += 1
            if days_without_period >= min_days_without_period:
                period_hasnt_started = True
    return stage_starts
                                                                
