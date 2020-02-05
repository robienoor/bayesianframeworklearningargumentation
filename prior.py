import numpy as np
from itertools import chain, combinations, product
import matplotlib.pyplot as plt
from utilities import calculateGroundedExtension, buildGraph


def generateGraphSpace(pos_args, neg_args):

    attacks = []

    # Generate the set of all possible attacks between the positive and negative arguments 
    for element in product(*[pos_args,neg_args]):
        attacks.append(element)
        
    for element in product(*[neg_args,pos_args]):
        attacks.append(element)

    # Generate the powerset of attacks. Each list in the powerset is a graph
    s = list(attacks)
    graphSpace =  list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

    return graphSpace


def buildGraph(attacks, pos_args, neg_args):

    noArgs = len(pos_args) + len(neg_args)
    all_args = pos_args + neg_args

    argMtx = np.zeros((noArgs,noArgs))

    if len(attacks) == 0:
        return argMtx
    else:
        for attack in attacks:
            all_args.index(attack[0])
            argMtx[all_args.index(attack[0]),
                all_args.index(attack[1])] = 1

    return argMtx


def calculateAggAttackScore(argMtx, pos_args, neg_args):

    all_args = pos_args + neg_args
    normGrade = 0

    for arg in all_args:

        if arg in pos_args:
            denominator = 2 * len(neg_args)
            min_grade = -1*(len(neg_args))
            grade_coeff = 1
        else:
            denominator = 2 * len(pos_args)
            min_grade = -1*(len(pos_args))
            grade_coeff = -1

        att_sum = np.sum(argMtx[all_args.index(arg),:])
        def_sum = np.sum(argMtx[:,all_args.index(arg)])
        arg_grade = att_sum - def_sum

        normGrade += (((arg_grade-min_grade) / denominator) * grade_coeff)

    return normGrade

def calculatePolScore(groundedExtension, pos_args, neg_args):

    all_args = pos_args + neg_args
    pol_score = 0
    for arg in groundedExtension:
        pol_score += 1 if all_args[arg] in pos_args else -1

    return pol_score

def calculateAggScores(graph_data, delta_att):

    aggs = [graph_data[0,1]]
    for graph_idx in range(1,len(graph_data)):
        if (graph_data[graph_idx,1] == graph_data[graph_idx-1,1]) and (graph_data[graph_idx,2] == graph_data[graph_idx-1,2]):
            aggs.append(aggs[graph_idx-1])
        else:
            aggs.append(aggs[graph_idx-1] - delta_att)

    return aggs


def getRatingToArgCoefficients(graph_data_complete, delta_att):
    # Split Graphs into Pos, Ntl and Neg sets and get agg coordinates
    agg_coordinates_y = [np.max(graph_data_complete[graph_data_complete[:,2] >0][:,3]),
                                np.min(graph_data_complete[graph_data_complete[:,2] >0][:,3]),
                                np.max(graph_data_complete[graph_data_complete[:,2] ==0][:,3]),
                                np.min(graph_data_complete[graph_data_complete[:,2] ==0][:,3]),
                                np.max(graph_data_complete[graph_data_complete[:,2] <0][:,3]),
                                np.min(graph_data_complete[graph_data_complete[:,2] <0][:,3])]

    if (np.max(graph_data_complete[graph_data_complete[:,2] >0][:,3]) == np.min(graph_data_complete[graph_data_complete[:,2] >0][:,3])):
        agg_coordinates_y[0] = agg_coordinates_y[0] + delta_att

    if (np.max(graph_data_complete[graph_data_complete[:,2] <0][:,3]) == np.min(graph_data_complete[graph_data_complete[:,2] <0][:,3])):
        agg_coordinates_y[-1] = agg_coordinates_y[-1] - delta_att

    agg_coordinates_x = [10,8,7,5,4,1]

    # Create Linear Model with Agg Coordinates
    ratingToArgCoefficients = np.polyfit(agg_coordinates_x, agg_coordinates_y, 2)

    return ratingToArgCoefficients


def computeGraphSummaries(pos_args, neg_args, delta_att):

    graphSpace = generateGraphSpace(pos_args, neg_args)
    arg_matrices = [buildGraph(g, pos_args, neg_args) for g in graphSpace]
    pol_scores = []
    attack_scores = []

    # Compute Pol Scores and Attack Scores
    for idx, g in enumerate(arg_matrices):
        pol_scores.append(calculatePolScore(calculateGroundedExtension(g), pos_args, neg_args))
        attack_scores.append(calculateAggAttackScore(g, pos_args, neg_args))

    # Sort by AttackScores and PolScores
    graph_data = np.array([list(range(len(graphSpace))), attack_scores, pol_scores]).T
    graph_data = graph_data[np.lexsort((graph_data[:,1], graph_data[:,2]))][::-1]

    aggs = calculateAggScores(graph_data , delta_att)

    # Place into larger structure
    graph_data_complete = np.zeros((len(aggs), 4))
    graph_data_complete[:,0:3] = graph_data
    graph_data_complete[:,-1] = aggs

    return graph_data_complete, arg_matrices

class prior:

    def __init__(self, pos_args, neg_args):
        self.pos_args = pos_args
        self.neg_args = neg_args
        self.delta_att = (1/(2*len(neg_args))) + (1/(2*len(pos_args))) 
        self.graph_data_complete, self.coefficients, arg_matrices = self.createPolynomialModel()

        # Sort the graphs by the agg values indices
        self.arg_matrices = [arg_matrices[int(i)] for i in self.graph_data_complete[:,0]]


    def createPolynomialModel(self):

        graph_data_complete, arg_matrices = computeGraphSummaries(self.pos_args, self.neg_args, self.delta_att)
        coefficients = getRatingToArgCoefficients(graph_data_complete, self.delta_att)

        return graph_data_complete, coefficients, arg_matrices

    def getDistribution(self, rating):

        self.rating = rating

        c = self.coefficients
        
        # Use the model to convert the rating to an agg score
        rating_to_agg_score = (c[0]*(rating**2)) + (c[1]*(rating)) + (c[2])

        aggs = self.graph_data_complete[:,-1]
        agg_distances = np.array([np.abs(agg - rating_to_agg_score) for agg in aggs])

        # max_agg_distance = np.max(agg_distances)
        # relative_agg_distances = [(max_agg_distance - agg_distance)**2 for agg_distance in agg_distances]
        
        # probabilties = [(relative_agg_distance / np.sum(relative_agg_distances)) for relative_agg_distance in relative_agg_distances]

        standardised_agg_distances = (1 / (1+(agg_distances**2)))
        probabilties = list(standardised_agg_distances / (np.sum(standardised_agg_distances)))

        return probabilties





