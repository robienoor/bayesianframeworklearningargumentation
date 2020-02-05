import numpy as np
from prior import prior
from itertools import product
from utilities import calculateGroundedExtension, buildGraph, convertArgMtrxToAttacks

def get_attack_type(arg_1, arg_2, attacks):

    if ((arg_1, arg_2) in attacks) and ((arg_2, arg_1) not in attacks):
        return '->'
    elif ((arg_1, arg_2) in attacks) and ((arg_2, arg_1) in attacks):
        return '<->'
    elif ((arg_1, arg_2) not in attacks) and ((arg_2, arg_1) not in attacks):
        return ' '
    else: 
        return '<-'

def similar_outcome(arg_1, arg_2, attacks):

    if ((arg_1, arg_2) in attacks) and ((arg_2, arg_1) not in attacks):
        return 1
    elif ((arg_1, arg_2) in attacks) and ((arg_2, arg_1) in attacks):
        return 0
    elif ((arg_1, arg_2) not in attacks) and ((arg_2, arg_1) not in attacks):
        return 0
    else: 
        return -1

def diff_similar_outcome_zzz(possible_conflicts, observation_data, graph, current_prior):

    graph_attacks = convertArgMtrxToAttacks(current_prior.pos_args, current_prior.neg_args, graph)

    sum_diffs = 0
    
    for possible_conflict in possible_conflicts:
        observation_outcome = similar_outcome(possible_conflict[0], possible_conflict[1], observation_data['attacks'])    
        graph_outcome = similar_outcome(possible_conflict[0], possible_conflict[1], graph_attacks)

        sum_diffs += 2- np.abs(observation_outcome - graph_outcome)

    return sum_diffs

def diff_similar_outcome(possible_conflicts, observation_data, graph, current_prior):

    graph_attacks = convertArgMtrxToAttacks(current_prior.pos_args, current_prior.neg_args, graph)

    sum_diffs = 0

    attack_types = ['->', '<->', ' ', '<-']

    attack_type_diffs = np.zeros((len(attack_types), len(attack_types)))
    attack_type_diffs[attack_types.index('->'), attack_types.index('<->')] = 1
    attack_type_diffs[attack_types.index('->'), attack_types.index(' ')] = 1
    attack_type_diffs[attack_types.index('->'), attack_types.index('<-')] = 2
    attack_type_diffs[attack_types.index('<->'), attack_types.index(' ')] = 1
    attack_type_diffs[attack_types.index('<->'), attack_types.index('<-')] = 1
    attack_type_diffs[attack_types.index(' '), attack_types.index('<-')] = 1

    attack_type_diffs = attack_type_diffs + attack_type_diffs.T - np.diag(np.diag(attack_type_diffs))
    sum_diffs = 0

    for possible_conflict in possible_conflicts:
        observation_attack_type = get_attack_type(possible_conflict[0], possible_conflict[1], observation_data['attacks'])
        graph_attack_type = get_attack_type(possible_conflict[0], possible_conflict[1], graph_attacks)

        diff = attack_type_diffs[attack_types.index(observation_attack_type), attack_types.index(graph_attack_type)]

        sum_diffs += (2- np.abs(diff)) / 2

    return sum_diffs


def groundedDiff(observation_data, graph_martix, observation_args, graph_space_args):
    graph_grounded = calculateGroundedExtension(graph_martix)
    observation_grounded = calculateGroundedExtension(observation_data['argMtx'])

    graph_grounded_args = [graph_space_args[g] for g in graph_grounded]
    observation_grounded_args = [observation_args[g] for g in observation_grounded]

    larger_grounded_extension = np.max([len(graph_grounded_args), len(observation_grounded_args)])
    
    if larger_grounded_extension == 0:
        return 1

    grounded_difference = set(graph_grounded_args).difference(set(observation_grounded_args)) | set(observation_grounded_args).difference(set(graph_grounded_args)) 

    groundedDiff = 1 / (1 + len(grounded_difference))

    return groundedDiff


def buildSimilarLiklihood(current_prior, observation_data, uniform_distribution_value):

    graph_space_args = current_prior.pos_args + current_prior.neg_args
    observation_args = observation_data['pos_args'] + observation_data['neg_args']

    pos_args_overlap = set(current_prior.pos_args).intersection(set(observation_data['pos_args']))
    neg_args_overlap = set(current_prior.neg_args).intersection(set(observation_data['neg_args']))

    possible_conflicts = [element for element in product(*[pos_args_overlap, neg_args_overlap])]

    diffs_sims = []
    diffs_grounded = []

    for graph in current_prior.arg_matrices:
        diffs_sims.append(diff_similar_outcome(possible_conflicts, observation_data, graph, current_prior))
        diffs_grounded.append(groundedDiff(observation_data, graph, observation_args,  graph_space_args))

    diffs_sims = np.array(diffs_sims)
    diffs_grounded = np.array(diffs_grounded)

    dis_total = diffs_sims + diffs_grounded
    normalising_constant = np.sum(dis_total)

    dis_total_distribution = dis_total / normalising_constant

    delta_args = len(list(set(observation_args).intersection(graph_space_args))) / np.max([len(observation_args), len(graph_space_args)])
    delta_rating = (10 - (np.abs(current_prior.rating - observation_data['rating']))) / 10

    liklihood_distribution = uniform_distribution_value - (delta_args*delta_rating*(uniform_distribution_value - dis_total_distribution))

    return liklihood_distribution


def buildMatchingLiklihood(current_prior, observation_data, uniform_distribution_value):
    observation_graph = observation_data['argMtx']

    matching_graph = 0
    for idx, graph in enumerate(current_prior.arg_matrices):
        if np.array_equal(observation_graph, graph):
            matching_graph = idx
            break

    aggs = current_prior.graph_data_complete[:,3]
    observation_agg = current_prior.graph_data_complete[matching_graph,3]

    agg_distances = 1 / (1 + np.abs(aggs - observation_agg))

    normalising_constant = np.sum(agg_distances)

    normalised_agg_distances = agg_distances / normalising_constant

    delta_rating = (10 - (np.abs(current_prior.rating - observation_data['rating']))) / 10

    liklihood_distribution = uniform_distribution_value - (delta_rating*(uniform_distribution_value - normalised_agg_distances))

    return liklihood_distribution

def classify_observation(graph_space_args, observation):
    all_observation_args = observation['pos_args'] + observation['neg_args']

    set_difference = list(set(graph_space_args).difference(set(all_observation_args)))
    observation_data = observation.copy()

    argMtx = buildGraph(observation['attacks'], observation['pos_args'], observation['neg_args'])
    observation_data['argMtx'] = argMtx

    if len(set_difference) == 0:
        return observation_data, 'matching' # passing either argMtx or returning observation. Feels like bad code

    else:
        return observation_data, 'similar'

class liklihood:

    def __init__(self, current_prior, observation):
        self.current_prior = current_prior
        self.uniform_distribution_value = 1 / len(current_prior.graph_data_complete)
        graph_space_args = current_prior.pos_args + current_prior.neg_args

        self.observation_data, self.observation_type = classify_observation(graph_space_args, observation)

    def buildLiklihoodDistribution(self):
        if self.observation_type == 'matching':
            self.liklihood_distribution = buildMatchingLiklihood(self.current_prior, self.observation_data, self.uniform_distribution_value)
        else:
            self.liklihood_distribution = buildSimilarLiklihood(self.current_prior, self.observation_data, self.uniform_distribution_value)

        return self.liklihood_distribution

pos_args = ['a','b']
neg_args = ['c']

p_G = prior(pos_args, neg_args)
prior_distribution = p_G.getDistribution(10)


observation = {}
observation['pos_args'] = ['a', 'b']
observation['neg_args'] = ['c']
observation['rating'] = 9
observation['attacks'] = [('a','c'), ('c','a')]

l = liklihood(p_G, observation)

l.buildLiklihoodDistribution()

print('h')






