import graphviz
from itertools import product
from prior import prior
from utilities import buildGraph

def generateGraph(pos_args, neg_args, arg_matrix):

    all_args = pos_args + neg_args

    d = graphviz.Digraph()
    d.node_attr['shape'] = 'square'
    on_colour = 'black'
    off_colour = 'white'
    all_attacks = list(product(*[pos_args,neg_args])) + list(product(*[neg_args,pos_args]))

    for element in all_attacks:

        arg1_idx = all_args.index(element[0])
        arg2_idx = all_args.index(element[1])

        if arg_matrix[arg1_idx, arg2_idx] ==1:
            d.edge(element[0],element[1], color=on_colour)
        else:
            d.edge(element[0],element[1], color=off_colour)

    return d


def create_graphs(pos_args, neg_args, arg_matrices):

    graphviz_dots = []

    location = '/Users/kawsarnoor/Desktop/PhD Work/code/bayesianframeworklearning/static/graphs/'
    graph_label = '_'.join(pos_args + neg_args)
    for idx, arg_mtx in enumerate(arg_matrices):
        d = generateGraph(pos_args, neg_args, arg_mtx)
        d.format = 'jpeg'
        d.graph_attr['rankdir'] = 'LR'
        d.render(location + graph_label + '_' + str(idx))
    return 

def create_observation_graph(pos_args, neg_args, attacks, observationNo):

    location = '/Users/kawsarnoor/Desktop/PhD Work/code/bayesianframeworklearning/static/graphs/'
    graph_label = 'observation_' + str(observationNo)

    arg_matrix = buildGraph(attacks, pos_args, neg_args)
    d = generateGraph(pos_args, neg_args, arg_matrix)
    d.format = 'jpeg'
    d.graph_attr['rankdir'] = 'LR'
    d.render(location + graph_label)

    return location + graph_label + '.jpeg'


    

