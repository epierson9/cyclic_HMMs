import matplotlib
from pylab import plot, title, show, imshow, savefig
from pomegranate import *
import numpy as np
from scipy.stats import norm, poisson, geom
import time
from IPython import embed
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from collections import Counter
import time
from constants_and_util import *
from multiprocessing import Process, Manager
import warnings
from traceback import print_exc
from scipy.misc import logsumexp

"""
Defines CyHMM and cluster CyHMM models. 
Indexing: 
i indexes over people / samples. 
j indexes over states
k indexes over symptoms
c indexes over clusters
t indexes over timesteps
"""

valid_distributions = ['poisson', 'geometric']
valid_emission_distributions = ['bernoulli_with_missing_data', 'normal_with_missing_data']

# zero_val and one_val are constants which are used to define the bernoulli_with_missing_data emission distribution, see below. 
# their actual values don't really matter; they're just a workaround because Pomegranate, the HMM package used
# has some trouble with discrete distributions. 
zero_val = 0 
one_val = 100

class emissionDistribution(object):
    """
    This class defines emission distributions. 
    It can define either Bernoulli distributions (for discrete data) or normal distributions (for continuous data). 
    """
    def __init__(self, emission_distribution_name, symptom_names, emission_parameters = None, samples = None, verbose = False):
        # checked. 
        self.emission_distribution_name = emission_distribution_name
        self.n_symptoms = len(symptom_names)
        self.symptom_names = symptom_names
        self.min_emission_prob = 1e-3
        self.verbose = verbose
        assert(self.emission_distribution_name in valid_emission_distributions)
        assert((emission_parameters is None) + (samples is None) == 1)
        if emission_parameters is not None:
            if self.verbose:
                print 'Initializing emission parameters from passed parameters'
            self.emission_parameters = emission_parameters
            self.initialize_distribution_from_params()
        else:
            if self.verbose:
                print 'Initializing emission parameters from samples'
            self.initialize_using_samples(samples)

    def add_random_noise_to_emission_parameters(self, emission_parameters, noise):
        """
        Adds noise to emission parameters. Don't want to initialize all emission parameters to be the same thing. Checked. 
        """
        for key in emission_parameters:
            emission_parameters[key] = np.array(emission_parameters[key])
            noise_matrix = np.ones(emission_parameters[key].shape) + noise * (np.random.random(emission_parameters[key].shape) - .5)
            emission_parameters[key] = emission_parameters[key] * noise_matrix
        return emission_parameters

    def initialize_using_samples(self, samples):  
        """
        this initializes the parameters + distributions. Checked. 
        """

        random_subset_of_samples = random.sample(range(len(samples)), max(1, min(2000, len(samples) / 2)))
        if self.emission_distribution_name == 'normal_with_missing_data':
            emission_parameters = {'p_missing':[], 'mu':[], 'sigma':[]}
            total_count_including_missing = 0
            non_missing_data = [[] for k in range(self.n_symptoms)]
            for i in random_subset_of_samples:
                X_i = np.array(samples[i])
                total_count_including_missing += len(X_i)
                data_available =  (X_i != missing_data_val)
                for k in range(self.n_symptoms):
                    non_missing_idxs = data_available[:, k]
                    non_missing_data[k] = non_missing_data[k] + list(X_i[non_missing_idxs, k])
            for k in range(self.n_symptoms):
                p_present = 1.*len(non_missing_data[k]) / total_count_including_missing
                emission_parameters['p_missing'].append(1 - p_present)
                emission_parameters['mu'].append(np.mean(non_missing_data[k]))
                emission_parameters['sigma'].append(np.std(non_missing_data[k]))

            for key in ['p_missing', 'sigma']: # clip these values because they should not be too small or less than 0. 
                emission_parameters[key] = np.clip(emission_parameters[key], self.min_emission_prob, np.inf)
            if len(samples) < 10:
                emission_parameters = self.add_random_noise_to_emission_parameters(emission_parameters, 1e-2)

        elif self.emission_distribution_name == 'bernoulli_with_missing_data':
            emission_parameters = {}
            pos_counts = [0 for k in range(self.n_symptoms)]
            total_count_including_missing = 0.0
            total_count = 0.0
            for i in random_subset_of_samples:
                total_count_including_missing += len(samples[i])
                for t in range(len(samples[i])):
                    if samples[i][t][0] != missing_data_val:
                        total_count += 1.0
                        for k in range(self.n_symptoms):
                            pos_counts[k] += samples[i][t][k]
            assert((total_count > 0) and (total_count_including_missing > 0))
            emission_parameters['p'] = np.clip(np.array(pos_counts) / total_count, self.min_emission_prob, 1 - self.min_emission_prob)
            emission_parameters['p_missing'] = np.clip(1 - total_count / total_count_including_missing, self.min_emission_prob, 1 - self.min_emission_prob)
            if len(samples) < 10:
                emission_parameters = self.add_random_noise_to_emission_parameters(emission_parameters, 1e-2)
        else:
            raise Exception("Not a valid emission distribution.")
        self.emission_parameters = emission_parameters
        self.initialize_distribution_from_params()

    def initialize_using_sufficient_statistics(self, summed_EX, summed_EX2, total_counts, total_counts_including_missing):
        """
        initializes the parameters + distribution using the sufficient statistics. Checked. 
        """
        if self.emission_distribution_name == 'normal_with_missing_data':
            self.emission_parameters['mu']= summed_EX / total_counts
            self.emission_parameters['sigma'] = np.sqrt(summed_EX2 / total_counts - self.emission_parameters['mu'] ** 2)
            assert((self.emission_parameters['sigma'] < 0).sum() == 0)
            self.emission_parameters['sigma'] = np.clip(self.emission_parameters['sigma'], self.min_emission_prob, np.inf)
            self.emission_parameters['p_missing'] = np.clip(1 - total_counts / total_counts_including_missing, self.min_emission_prob, 1)
        if self.emission_distribution_name == 'bernoulli_with_missing_data':
            assert(np.abs(total_counts[0] - total_counts[1]) < 1e-6)
            assert(np.abs(total_counts_including_missing[0] - total_counts_including_missing[1]) < 1e-6)
            self.emission_parameters['p_missing'] = np.clip(1 - 1.*total_counts[0] / total_counts_including_missing[0], self.min_emission_prob, (1 - self.min_emission_prob))
            self.emission_parameters['p'] = np.clip(np.array(summed_EX / total_counts), self.min_emission_prob, (1 - self.min_emission_prob))
        self.initialize_distribution_from_params()

    def make_fake_discrete_distribution(self, centers, ps):
        """
        Pomegranate handles discrete distributions a little oddly: this is a workaround. 
        Small helper method for the method below. 
        The density of a draw from the center_i of this distribution will be ps_i. 
        """
        n_components = len(centers)
        assert([(p_i > 0) and (~np.isnan(p_i)) for p_i in ps])
        components = [UniformDistribution(centers[i] - 1 / (2 * ps[i]), centers[i] + 1 / (2 * ps[i])) for i in range(n_components)]
        model = GeneralMixtureModel(components, weights = np.array([1. / n_components for i in range(n_components)]))
        return model

    def create_bernoulli_distribution_with_missing_data(self, p, p_missing):
        """
        creates a fake Bernoulli distribution with missing data. Ie, with probability p_missing all data is missing. 
        Otherwise, it is on with probability p. 
        Confirmed that this (somewhat hacky) setup yields log_probabilities which agree with what we would expect (up to a factor of 3 ^ n_symptoms, 
        which should just add a constant term for everything.)
        """
        n_symptoms = len(p)
        mixture_models = []
        for k in range(n_symptoms):
            model_for_symptom = self.make_fake_discrete_distribution(centers = [missing_data_val, one_val, zero_val], 
                                                                ps = [p_missing ** (1. / n_symptoms), 
                                                                      p[k] * (1 - p_missing) ** (1. / n_symptoms), 
                                                                      (1 - p[k]) * (1 - p_missing) ** (1. / n_symptoms)])
            mixture_models.append(model_for_symptom)
        return IndependentComponentsDistribution(mixture_models) 

    def initialize_distribution_from_params(self): 
        """
        this initializes the distributions (ie, the Pomegranate objects) using the parameters. 
        Checked.
        """
        if self.emission_distribution_name == 'normal_with_missing_data':
            mixture_models = []
            for k in range(self.n_symptoms):
                p_missing = self.emission_parameters['p_missing'][k]
                mu = self.emission_parameters['mu'][k]
                sigma = self.emission_parameters['sigma'][k]
                mixture_models.append(GeneralMixtureModel( [NormalDistribution(mu, sigma), UniformDistribution(missing_data_val - .5, missing_data_val + .5)], weights=np.array([1 - p_missing, p_missing]))) # the width of the missing data distribution (a uniform distribution) should not matter for the purposes of the E-step, since it is the same for all states. But it should be .5. 
            distribution = IndependentComponentsDistribution(mixture_models)
        
        if self.emission_distribution_name == 'bernoulli_with_missing_data':
            distribution = self.create_bernoulli_distribution_with_missing_data(self.emission_parameters['p'], self.emission_parameters['p_missing'])

        self.emission_distribution = distribution

    def get_state_emission_params(self):
        """
        Returns emission parameters. 
        Checked. 
        """
        symptoms_to_probabilities = {} # keys are short symptom names, values are emission parameters. 
        for k in range(self.n_symptoms):
            symptom_distribution = self.emission_distribution.parameters[0][k]
            if self.emission_distribution_name == 'normal_with_missing_data': # need to extract mu, sigma, p_missing. 
                symptoms_to_probabilities[self.symptom_names[k] +'_p_missing'] = np.exp(symptom_distribution.weights)[1]
                symptoms_to_probabilities[self.symptom_names[k] +'_mean'] = symptom_distribution.distributions[0].parameters[0]
                symptoms_to_probabilities[self.symptom_names[k] +'_std'] = symptom_distribution.distributions[0].parameters[1]
            if self.emission_distribution_name == 'bernoulli_with_missing_data':
                symptoms_to_probabilities[self.symptom_names[k] + '_p'] = self.emission_parameters['p'][k]

        if self.emission_distribution_name == 'bernoulli_with_missing_data':
            symptoms_to_probabilities['p_missing'] = self.emission_parameters['p_missing']
        return symptoms_to_probabilities

class cyclic_HMM(object): 
    """
    This class is the main CyHMM class. 
    Checked. 
    """
    def __init__(self, n_states, state_duration_means, state_duration_scales, duration_distribution_name, emission_distribution_name, max_duration, emission_parameters = None, samples = None, symptom_names = None, verbose = True, min_iterations = 0):
        self.n_states = n_states
        assert(duration_distribution_name in valid_distributions)
        assert(emission_distribution_name in valid_emission_distributions)
        assert(self.n_states == len(state_duration_means) == len(state_duration_scales))
        assert((emission_parameters is not None) or (samples is not None))
        self.duration_distribution_name = duration_distribution_name
        self.emission_distribution_name = emission_distribution_name
        self.state_duration_means = state_duration_means
        self.state_duration_scales = state_duration_scales # we do not actually use scales at present (it's a shape parameter for the duration distribution) but it might be useful if you wanted to fit more complex duration distributions. 
        self.max_duration = max_duration
        self.symptom_names = symptom_names
        self.verbose = verbose
        self.min_iterations = min_iterations
        self.n_symptoms = len(self.symptom_names)
        self.samples = samples
        self.emission_distributions = []
        for j in range(self.n_states):
            self.emission_distributions.append(emissionDistribution(self.emission_distribution_name, self.symptom_names, emission_parameters, samples, verbose = self.verbose))

        self.initialize_hmm()

    def initialize_hmm(self): 
        """
        Initializes the HMM. 
        checked. 
        """
        self.hmm = HiddenMarkovModel()
        assert(self.duration_distribution_name in valid_distributions)
        self.initialize_states()
        self.initialize_transition_matrix()
        self.hmm.bake()


    def sample(self, length = 50): 
        """
        Draws a random sample. Checked. 
        """
        return self.hmm.sample(length = length)

    def model_has_converged(self, convergence_tol = 1e-2): 
        """
        Assesses whether the model has converged using the given convergence tolerance. Checked. 
        """
        if len(self.all_params_by_iteration) < 2:
            return False
        old = np.array(self.all_params_by_iteration[-2])
        new = np.array(self.all_params_by_iteration[-1])
        feature_names = self.all_params_by_iteration[-2].index
        relative_change = np.abs((old - new) / np.clip(np.abs(new), 1e-6, np.inf))
        absolute_change = np.abs(old - new)
        overall_change = np.minimum(relative_change, absolute_change)
        max_row, max_col = np.where(overall_change == overall_change.max())
        if self.verbose:
            print 'Testing convergence; maximum change is %2.3f in %s' % (overall_change.max(), feature_names[max_row[0]])
        if (overall_change.max() < convergence_tol):
            return True
        return False

    def fit_helper(self, samples, states_to_idxs, return_dict, process_idx): 
        """
        Parallelizes the E-step so it can use multiple processes. 
        checked for both Normal and Bernoulli. 
        """
        all_transition_counts = 0
        summed_EX = np.zeros([self.n_states, self.n_symptoms]) # sum non-missing values of X over all samples, weighting by probability you're in a state. 
        summed_EX2 = np.zeros([self.n_states, self.n_symptoms]) # sum non-missing values of X^2 over all samples, weighting by probability you're in a state. 
        total_counts = np.zeros([self.n_states, self.n_symptoms]) # sum the probability you're in a state over all (non-missing) samples. 
        total_counts_including_missing = np.zeros([self.n_states, self.n_symptoms]) # sum the probability you're in a state over all (non-missing) samples. 
        for sample in samples:
            if self.emission_distribution_name == 'bernoulli_with_missing_data':
                # small hack -- need to remap values to the values the distribution expects. 
                mapper_d = {0:zero_val, 1:one_val, missing_data_val:missing_data_val}
                transition_counts_i, log_state_probs_i = self.hmm.forward_backward([[mapper_d[a] for a in b] for b in sample])
            else:
                transition_counts_i, log_state_probs_i = self.hmm.forward_backward(sample)
            state_probs_i = np.exp(log_state_probs_i)
            non_missing_data = (np.array(sample) != missing_data_val)
            for j in range(self.n_states):
                state_idxs = states_to_idxs[j]
                for k in range(self.n_symptoms):
                    non_missing_idxs = np.flatnonzero(non_missing_data[:, k])
                    prob_in_state = state_probs_i[:, state_idxs].sum(axis = 1)
                    total_counts_including_missing[j][k] += sum(prob_in_state) 
                    prob_in_state = prob_in_state[non_missing_idxs]
                    non_missing_symptom_vals = np.array(sample)[non_missing_idxs, k]
                    total_counts[j][k] += sum(prob_in_state)
                    summed_EX[j][k] += sum(prob_in_state * non_missing_symptom_vals)
                    summed_EX2[j][k] += sum(prob_in_state * (non_missing_symptom_vals ** 2))
            all_transition_counts = all_transition_counts + transition_counts_i

        return_dict[process_idx] = [all_transition_counts, summed_EX, summed_EX2, total_counts, total_counts_including_missing]

    def fit(self, max_iterations = 50, n_processes = 40, **kwargs): 
        """
        Main fit method. 
        checked for both Normal + Bernoulli.
        """
        self.all_params_by_iteration = []
        # what state indices correspond to each state? Warning: if you have SILENT STATES before end it will break. We do not. 
        states_to_idxs = {}
        for j in range(self.n_states):
            states_to_idxs[j] = [idx for idx in range(len(self.hmm.states)) if 'state_%i' % j in self.hmm.states[idx].name] 
        assert(self.hmm.states[-2].name == 'None-start')
        assert(self.hmm.states[-1].name == 'None-end')
        n_processes_to_use = min(n_processes, len(self.samples))
        process_cutpoints = np.linspace(0, len(self.samples), n_processes_to_use + 1).astype(int)
        manager = Manager()
        return_dict = manager.dict()
        for iteration in range(max_iterations):
            t0 = time.time()
            processes = []

            for process_idx in range(n_processes_to_use):
                args = [self.samples[process_cutpoints[process_idx]:process_cutpoints[process_idx + 1]], states_to_idxs, return_dict, process_idx]
                p = Process(target=self.fit_helper, args=tuple(args))
                processes.append(p)
            for process in processes:
                process.start()
            for process in processes:
                process.join()
            for process_idx in range(n_processes_to_use):
                if process_idx == 0:
                    all_transition_counts, summed_EX, summed_EX2, total_counts, total_counts_including_missing = return_dict[process_idx]
                else:
                    all_transition_counts_i, summed_EX_i, summed_EX2_i, total_counts_i, total_counts_including_missing_i = return_dict[process_idx]
                    all_transition_counts += all_transition_counts_i
                    summed_EX += summed_EX_i
                    summed_EX2 += summed_EX2_i
                    total_counts += total_counts_i
                    total_counts_including_missing += total_counts_including_missing_i
            
            if self.verbose:
                print 'Time to do all model fitting', time.time() - t0
            duration_means = []
            duration_scales = []
            for j in range(self.n_states):
                self.emission_distributions[j].initialize_using_sufficient_statistics(summed_EX[j, :], summed_EX2[j, :], total_counts[j, :], total_counts_including_missing[j, :])
                duration_mean, duration_scale = self.get_state_duration_params(j, transition_matrix_counts = all_transition_counts)
                duration_means.append(duration_mean)
                duration_scales.append(duration_scales)

            # Basically we run one iteration and then create a new HMM. This is pretty fast. 
            self.state_duration_means = duration_means
            self.state_duration_scales = duration_scales
            self.initialize_hmm()     
            if self.verbose:
                print 'After iteration %i (total time: %2.3f) parameters are' % (iteration, time.time() - t0)
            all_params = self.get_all_params()
            assert(np.isnan(np.array(all_params)).sum() == 0)
            converged = self.model_has_converged()
            self.all_params_by_iteration.append(all_params)
            self.iteration = iteration

            if self.verbose:
                print all_params
                print all_params.loc['duration_mean'].sum() + len(all_params.columns), 'total cycle length'
                self.compute_cycle_lengths_using_viterbi()
            if converged and iteration >= self.min_iterations:
                if self.verbose:
                    print 'Converged after iteration', iteration
                break
                
    def get_viterbi_path(self, X_i): 
        """
        Returns the Viterbi path (the most likely sequence of hidden states) for a given sample. Checked. 
        """
        if self.emission_distribution_name != 'bernoulli_with_missing_data':
            viterbi_path = self.hmm.viterbi(X_i)[1]
        else:
            mapper_d = {0:zero_val, 1:one_val, missing_data_val:missing_data_val}
            viterbi_path = self.hmm.viterbi([[mapper_d[a] for a in b] for b in  X_i])[1]
        
        viterbi_path = np.array([self.turn_viterbi_state_to_int(Z_t[1].name) for Z_t in viterbi_path][1:])
        return viterbi_path
                                
    def compute_cycle_lengths_using_viterbi(self): 
        """
        Checked. Returns the median and mean cycle length for a small subset of samples. 
        If you print this while fitting, helps confirm model is heading in the right direction. 
        """
        all_inferred_cycle_lengths = []
        for X_i in random.sample(self.samples, min(len(self.samples), 300)):
            viterbi_path = self.get_viterbi_path(X_i)
            all_inferred_cycle_lengths += list(get_cycle_lengths(extract_stage_starts(viterbi_path == 0)))
        print 'Mean inferred cycle length using viterbi: %2.1f; median: %2.1f' % (np.mean(all_inferred_cycle_lengths), np.median(all_inferred_cycle_lengths))
    
    def turn_viterbi_state_to_int(self, Z_i): 
        """ 
        small helper method: return an int rather than the state name (a string). Checked. 
        """
        Z_i = Z_i.split('_duration')[0]
        if 'start' in Z_i:
            return -1
        return int(Z_i.replace('state_', ''))

    def get_duration_pdf(self, duration = None, state = None): 
        """
        checked. Computes probability of remaining in state for given duration. 
        Scale here should be standard deviation. 
        """
        if self.duration_distribution_name == 'poisson':
            return poisson.pmf(duration, self.state_duration_means[state])
        if self.duration_distribution_name == 'geometric':
            return geom.pmf(duration, 1. / self.state_duration_means[state])

    def initialize_states(self): 
        """
        checked. Initializes the states.
        """
        self.states = []
        self.state_indices = []
        for j in range(self.n_states):            
            for t in range(self.max_duration, -1, -1):
                self.state_indices.append([j, t])
                state = State(self.emission_distributions[j].emission_distribution, name = 'state_%i_duration_%i' % (j, t))
                self.states.append(state)
        self.hmm.add_states(self.states)

    def initialize_transition_matrix(self): 
        """
        checked. Initializes transition matrix.
        """
        # The reason this uses self.state_indices and self.states is that self.hmm.states does not exist until we bake the model. self.state_indices and self.states are lists which do not include start and end states. 
        for idx in range(len(self.state_indices)):
            duration = self.state_indices[idx][1]
            j = self.state_indices[idx][0]
            self.hmm.add_transition(self.hmm.start, self.states[idx], self.get_duration_pdf(duration = duration, state = j))

            if duration != 0: # keep timer counting down. 
                self.hmm.add_transition(self.states[idx], self.states[idx + 1], 1)
            else: # if we're at duration zero, fan out to all next states. 
                if j < self.n_states - 1:
                    next_j = j + 1
                    idx2 = idx + 1 # idx2 denotes the index of the states to transition to in the state_indices table. 
                else:
                    next_j = 0
                    idx2 = 0
                while idx2 < len(self.state_indices) and self.state_indices[idx2][0] == next_j:
                    duration = self.state_indices[idx2][1]
                    p = self.get_duration_pdf(duration = duration, state = next_j)
                    self.hmm.add_transition(self.states[idx], self.states[idx2], p)
                    idx2 += 1

    def get_state_duration_params(self, j, transition_matrix_counts = None): 
        """
        Checked. j denotes state here. 
        If transition_matrix_counts is not None, computes the mean and std in state duration based on transition counts from previous state. 
        """
        previous_j = j - 1 if j > 0 else self.n_states - 1
        durations_to_probabilities = {} # dictionary where key is duration, value is probability of that duration. 
        for idx, state in enumerate(self.hmm.states):
            if state.name == 'state_%i_duration_0' % previous_j:
                if transition_matrix_counts is None:
                    transitions = self.hmm.dense_transition_matrix()[idx,:]
                else:
                    transitions = transition_matrix_counts[idx, :] / transition_matrix_counts[idx, :].sum()
                assert(transitions.shape[0] == len(self.hmm.states))
                for idx2 in range(len(transitions)):
                    if transitions[idx2] > 1e-8:
                        assert(self.hmm.states[idx2].name.split('_duration')[0] == ('state_%i' % j))
                        duration = int(self.hmm.states[idx2].name.split('duration_')[1])
                        durations_to_probabilities[duration] = transitions[idx2]

        durations = np.array(durations_to_probabilities.keys())
        ps = np.array(durations_to_probabilities.values())
        duration_mean = sum(durations * ps) / sum(ps)
        duration_std = np.sqrt(np.average((durations - duration_mean)**2, weights=ps))

        return duration_mean, duration_std
        
    def get_all_params(self): 
        """
        checked. Returns the parameters for the model.
        """
        params = pd.DataFrame()
        for j in range(self.n_states):
            state_params = self.emission_distributions[j].get_state_emission_params()
            duration_mean, duration_std = self.get_state_duration_params(j)
            state_params['duration_mean'] = [duration_mean]
            state_params['duration_std'] = [duration_std]
            params = pd.concat([params, pd.DataFrame(state_params)])
        params = params.transpose()
        params.columns = range(self.n_states)
        
        return params

    def log_probability(self, X_i):
        """
        Checked. Returns the log probability of a given individual sample. 
        """
        if self.emission_distribution_name == 'bernoulli_with_missing_data':
            mapper_d = {0:zero_val, 1:one_val, missing_data_val:missing_data_val}
            ll = self.hmm.log_probability([[mapper_d[a] for a in b] for b in X_i])
        else:
            ll = self.hmm.log_probability(X_i)
        return ll

class clusterHMM(object): 
    """
    This class wraps around the main CyHMM class and performs clustering. Checked. 
    """
    def __init__(self, n_states, n_clusters, samples, symptom_names, duration_distribution_name, emission_distribution_name, hypothesized_duration, n_samples_to_use_in_clustering = 50):
        self.n_states = n_states
        self.n_symptoms = len(samples[0][0])
        self.n_clusters = n_clusters
        self.symptom_names = symptom_names
        self.duration_distribution_name = duration_distribution_name
        self.hypothesized_duration = hypothesized_duration
        self.samples = samples
        self.cluster_models = []
        self.clusterings_by_iteration = []
        self.emission_distribution_name = emission_distribution_name
        print 'Fitting clustering model with n_states = %i, n_clusters = %i, n_samples = %i, n_symptoms = %i' % (self.n_states, self.n_clusters, len(self.samples), self.n_symptoms)
        self.n_samples_to_use_in_clustering = n_samples_to_use_in_clustering
        self.produce_initial_clustering()


    def produce_initial_clustering(self, n_processes_to_use = 20):
        """ 
        Initialize clustering.
        """
        random_samples = random.sample(range(len(self.samples)), self.n_samples_to_use_in_clustering)
        n_processes_to_use = min(len(random_samples), n_processes_to_use)
        process_cutpoints = np.linspace(0, len(random_samples), n_processes_to_use + 1).astype(int)
        manager = Manager()
        return_dict = manager.dict()
        processes = []
        for process_idx in range(n_processes_to_use):
            args = [random_samples[process_cutpoints[process_idx]:process_cutpoints[process_idx + 1]], return_dict, process_idx]
            p = Process(target=self.produce_initial_clustering_helper, args=tuple(args))
            processes.append(p)
        for process in processes:
            process.start()
        for process in processes:
            process.join()

        for key in sorted(return_dict.keys()):
            if key == 0:
                all_log_probs = return_dict[key]
            else:
                all_log_probs = np.hstack([all_log_probs, return_dict[key]])
        for row_idx in range(len(all_log_probs)):
            if all_log_probs[row_idx].std() > 0:
                all_log_probs[row_idx] = (all_log_probs[row_idx] - all_log_probs[row_idx].mean()) / all_log_probs[row_idx].std()
            else:
                all_log_probs[row_idx] = (all_log_probs[row_idx] - all_log_probs[row_idx].mean())
        assert((np.isnan(all_log_probs) | np.isinf(all_log_probs)).sum() == 0)
        kmeans_fitter = KMeans(n_clusters = self.n_clusters)
        

        kmeans_fitter.fit(all_log_probs)
        M = all_log_probs[np.argsort(kmeans_fitter.labels_)]
        #imshow(M, interpolation = 'nearest', aspect='auto')
        print "Done with initial clustering"
        print Counter(kmeans_fitter.labels_)
        self.clustering = kmeans_fitter.labels_ 


    def produce_initial_clustering_helper(self, random_samples, return_dict, process_idx): 
        """
        Small helper method to parallelize initialization. 
        """
        all_one_sample_models = []

        for i, idx in enumerate(random_samples):
            model = fit_cyhmm_model(self.n_states, 
                                    [self.samples[idx]], 
                                    self.symptom_names, 
                                    hypothesized_duration = self.hypothesized_duration,
                                    emission_distribution_name = self.emission_distribution_name,
                                    duration_distribution_name = self.duration_distribution_name, 
                                    verbose = False)
            all_one_sample_models.append(model)
        log_probabilities_per_timestep = []
        for i in range(len(self.samples)):
            log_probabilities_per_timestep_i = []
            for model_idx in range(len(all_one_sample_models)):
                log_probabilities_per_timestep_i.append(all_one_sample_models[model_idx].log_probability(self.samples[i]) / len(self.samples[i]))
            log_probabilities_per_timestep.append(log_probabilities_per_timestep_i)
        return_dict[process_idx] = np.array(log_probabilities_per_timestep)
        
    def fit(self, max_clustering_iterations = 50): # checked. 
        iteration = 0
        self.clusterings_by_iteration.append(self.clustering)
        while iteration < max_clustering_iterations:
            
            print '\n\n\n\n****Clustering iteration', iteration
            print 'Refitting clusters'
            t0 = time.time()
            self.refit_clusters()
            print 'Time to refit clusters', time.time() - t0
            print '****Reassigning samples'
            t0 = time.time()
            self.reassign_to_clusters()
            print 'Time to reassign to clusters', time.time() - t0
            print 'Cluster counts', Counter(self.clustering)
            self.print_properties_of_clustering(compute_symptom_probability_by_person = False)
            iteration += 1
            self.clusterings_by_iteration.append(self.clustering)
            if len(self.clusterings_by_iteration) >= 2:
                n_points_changed = 1.*(np.array(self.clusterings_by_iteration[-2]) != np.array(self.clusterings_by_iteration[-1])).sum()
                print 'Number of points that changed clustering: %i / %i' % (n_points_changed, len(self.clustering))
                if n_points_changed / len(self.clustering) < .01:
                    print "clustering converged!"
                    break

    def print_properties_of_clustering(self, compute_symptom_probability_by_person = False, compute_cycle_length = True):
        symptom_frequencies_df = []
        print 'Printing properties of clustering!!!'
        for c in sorted(list(set(self.clustering))):
            cycle_length = 0
            if len(self.cluster_models) > 0:
                print self.cluster_models[c].get_all_params()
            if compute_cycle_length:
                for j in range(self.n_states):
                    duration_mean, duration_std = self.cluster_models[c].get_state_duration_params(j)
                    
                    cycle_length += duration_mean + 1
            cluster_idxs = np.flatnonzero(self.clustering == c)
            cluster_samples = [self.samples[i] for i in cluster_idxs]
            tot_samples = np.zeros([self.n_symptoms,])
            summed_vals = np.zeros([self.n_symptoms,])
            tot_samples_including_missing = np.zeros([self.n_symptoms,])
            for sample in cluster_samples:
                sample = np.array(sample)
                if compute_symptom_probability_by_person:
                    tot_samples += 1
                    summed_vals += (np.array(sample).sum(axis = 0) > 0)
                else:
                    for k in range(self.n_symptoms):
                        non_missing_idxs = (sample[:, k] != missing_data_val) & (~np.isnan(sample[:, k]))
                        summed_vals[k] += sample[non_missing_idxs, k].sum()
                        tot_samples[k] += non_missing_idxs.sum()
                        tot_samples_including_missing[k] += len(sample)

            p = summed_vals / tot_samples
            p_missing = 1 - tot_samples / tot_samples_including_missing
            symptom_frequencies = dict(zip(self.symptom_names, p))

            if self.emission_distribution_name == 'bernoulli_with_missing_data':
                symptom_frequencies['symptoms_logged_per_day'] = summed_vals.sum() / (tot_samples[0])
                symptom_frequencies['p_missing'] = p_missing[0]
            else:
                symptom_frequencies.update(dict(zip(['missing_%s' % a for a in self.symptom_names], p_missing)))
            symptom_frequencies['n_people'] = len(cluster_idxs)
            symptom_frequencies['cycle_length'] = cycle_length
            symptom_frequencies_df.append(symptom_frequencies)
        symptom_frequencies_df = pd.DataFrame(symptom_frequencies_df)
        if compute_symptom_probability_by_person:
            print 'Symptom frequencies by cluster (by person)'
            if self.emission_distribution_name != 'bernoulli_with_missing_data':
                print "WARNING: THIS DOES NOT MAKE SENSE FOR NON-BERNOULLI EMISSIONS."
        else:
            print 'Symptom frequencies by cluster (by timestep)'
        print symptom_frequencies_df.transpose()
        return symptom_frequencies_df

    def refit_clusters(self): 
        """
        Refit the clusters given the new cluster assignments.
        """
        new_cluster_models = []
        for c in sorted(list(set(self.clustering))):
            cluster_idxs = np.flatnonzero(self.clustering == c)
            print 'Refitting cluster %i with %i samples' % (c, len(cluster_idxs))
            cluster_samples = [self.samples[i] for i in cluster_idxs]
            if len(self.cluster_models) == 0: # have not fit models yet. 
                print 'Fitting initial clustering model %i' % c
                model = fit_cyhmm_model(self.n_states, 
                                        cluster_samples, 
                                        self.symptom_names, 
                                        hypothesized_duration = self.hypothesized_duration,
                                        duration_distribution_name = self.duration_distribution_name,
                                        emission_distribution_name = self.emission_distribution_name,
                                        verbose = True, 
                                        max_iterations = 20)
                
                print 'Model converged after %i iterations' % model.iteration
                print model.get_all_params()
            else:
                print 'Initializing model %i from prior optimized state. Starting parameters' % c
                model = self.cluster_models[c]
                model.samples = cluster_samples
                print model.get_all_params()
                model.fit(max_iterations = 5) 
                model.samples = None
                print 'Model converged after %i iterations' % model.iteration
                print model.get_all_params()
            new_cluster_models.append(model)

        self.cluster_models = new_cluster_models

    def reassign_to_clusters(self): 
        """
        Checked. Reassigns samples to clusters.
        """
        clusters_to_ps = Counter(self.clustering)
        for c in clusters_to_ps:
            clusters_to_ps[c] = 1.*clusters_to_ps[c] / len(self.clustering)
        self.clustering = []
        for sample in self.samples:
            log_probabilities = []
            for c, cluster_model in enumerate(self.cluster_models):
                log_probabilities.append(cluster_model.log_probability(sample) + np.log(clusters_to_ps[c]))
            self.clustering.append(np.argmax(log_probabilities))
    
    def log_probability(self, X_i):
        """
        Checked. Computes the log probability for a given sample. 
        """
        clusters_to_ps = Counter(self.clustering)
        for c in clusters_to_ps:
            clusters_to_ps[c] = 1.*clusters_to_ps[c] / len(self.clustering)
        log_probabilities = []
        for c, cluster_model in enumerate(self.cluster_models):
            log_probabilities.append(cluster_model.log_probability(X_i) + np.log(clusters_to_ps[c]))
        return logsumexp(log_probabilities)

    def get_viterbi_path(self, X_i):
        """
        Computes the viterbi path for a sample. 
        """
        cluster_ps = Counter(self.clustering)
        for c in cluster_ps:
            cluster_ps[c] = 1. * cluster_ps[c] / len(self.clustering)
        log_probs = []
        for c in range(self.n_clusters):
            log_probs.append(self.cluster_models[c].log_probability(X_i) + np.log(cluster_ps[c]))
        cluster_assignment = np.argmax(log_probs)
        viterbi_path = self.cluster_models[cluster_assignment].get_viterbi_path(X_i)
        return viterbi_path

def fit_cyhmm_model(n_states, 
                    samples, 
                    symptom_names, 
                    hypothesized_duration, 
                    duration_distribution_name, 
                    emission_distribution_name, 
                    verbose = True, 
                    n_processes = 5, 
                    min_iterations = 0, 
                    max_iterations = 50):
    """
    fits a CyHMM on a set of samples. n_states is the number of states. 
    samples is an list of samples where each element is a n_timesteps x n_symptoms matrix. 
    symptom_names is a list of symptom_names. 
    """
    model = cyclic_HMM(n_states = n_states, 
                       state_duration_means = [float(hypothesized_duration) / n_states - 1 for i in range(n_states)],
                       state_duration_scales = [5 for i in range(n_states)], 
                       duration_distribution_name = duration_distribution_name, 
                       emission_distribution_name = emission_distribution_name, 
                       max_duration = 25, 
                       samples = samples, 
                       symptom_names = symptom_names, 
                       verbose = verbose, 
                       min_iterations = min_iterations)
    model.fit(max_iterations = max_iterations, n_processes = n_processes)
    model.samples = None  
    return model

def fit_clustering_model(n_states, 
                         n_clusters, 
                         samples, 
                         symptom_names, 
                         duration_distribution_name, 
                         emission_distribution_name, 
                         hypothesized_duration, 
                         n_samples_to_use_in_clustering = 50, 
                         max_clustering_iterations = 20):
    """
    fits a clustering on a set of samples. n_states is the number of states. 
    samples is an list of samples where each element is a n_timesteps x n_symptoms matrix. 
    symptoms is a list of symptom_names. 
    """                            
    model = clusterHMM(n_states, 
                       n_clusters, 
                       samples, 
                       symptom_names,  
                       duration_distribution_name = duration_distribution_name,
                       emission_distribution_name = emission_distribution_name,
                       n_samples_to_use_in_clustering = n_samples_to_use_in_clustering, 
                       hypothesized_duration = hypothesized_duration)
    model.fit(max_clustering_iterations = max_clustering_iterations)
    model.samples = None
    return model




