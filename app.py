from flask import Flask, render_template, jsonify, request
import json
import numpy as np
from prior import prior
from liklihood import liklihood
from posterior import posterior
import graph_image_generator

app = Flask(__name__)

@app.route('/generatePosteriorDistributionWithObsevation')
def generatePosteriorDistributionWithObsevation():

    observation = {}

    pos_args_string = request.args.get('pos_args')
    observation['pos_args'] = json.loads(pos_args_string)

    neg_args_string = request.args.get('neg_args')
    observation['neg_args'] = json.loads(neg_args_string)

    observation['rating'] = int(request.args.get('rating'))
    observation['observationNo'] = int(request.args.get('observationNo'))
    
    attacks_string = request.args.get('attacks')
    attacks_raw = json.loads(attacks_string)
    observation['attacks'] = [tuple(attack) for attack in attacks_raw]

    currentPriorString = request.args.get('currentPrior')
    currentPrior = np.array(json.loads(currentPriorString))

    # Will need to rebuild all of the important graph space data as this is needed for the liklihood construction
    graphSpaceSummaryString = request.args.get('graphSpaceSummary')
    graphSpaceSummary = json.loads(graphSpaceSummaryString)

    p_G = prior(graphSpaceSummary['pos_args'], graphSpaceSummary['neg_args'])
    p_G.rating = graphSpaceSummary['rating'] # This is possibly reckless coding

    p_G_T = liklihood(p_G, observation)

    liklihood_distribution = p_G_T.buildLiklihoodDistribution()

    p_T_G = posterior(currentPrior, liklihood_distribution)

    posterior_distribution = p_T_G.buildPosteriorDistribution()

    distributions = {}
    distributions['liklihood_distribution'] = list(liklihood_distribution)
    distributions['posterior_distribution'] = list(posterior_distribution)

    return jsonify(distributions)



@app.route('/generateObservationGraph')
def generateObservationGraph():

    pos_args_string = request.args.get('pos_args')
    pos_args = json.loads(pos_args_string)

    neg_args_string = request.args.get('neg_args')
    neg_args = json.loads(neg_args_string)

    rating = int(request.args.get('rating'))
    observationNo = int(request.args.get('observationNo'))
    
    attacks_string = request.args.get('attacks')
    attacks = json.loads(attacks_string)
 
    graph_location = graph_image_generator.create_observation_graph(pos_args, neg_args, attacks, observationNo)

    return jsonify(graph_location)


@app.route('/generateGraphSpacewithPrior')
def generateGraphSpacewithPrior():

    pos_args_string = request.args.get('pos_args')
    pos_args = json.loads(pos_args_string)

    neg_args_string = request.args.get('neg_args')
    neg_args = json.loads(neg_args_string)

    rating = int(request.args.get('rating'))

    # generate the graph images
    p_G = prior(pos_args, neg_args)
    graph_image_generator.create_graphs(pos_args, neg_args, p_G.arg_matrices)

    # generate the prior distribution
    prior_distribution = p_G.getDistribution(rating)

    return jsonify(prior_distribution)

@app.route('/generateGraphSpacewithUniformPrior')
def generateGraphSpacewithUniformPrior():
    
    pos_args_string = request.args.get('pos_args')
    pos_args = json.loads(pos_args_string)

    neg_args_string = request.args.get('neg_args')
    neg_args = json.loads(neg_args_string)

    rating = int(request.args.get('rating'))

    # generate the graph images
    p_G = prior(pos_args, neg_args)
    graph_image_generator.create_graphs(pos_args, neg_args, p_G.arg_matrices)

    # generate the uniform distribution
    prior_distribution = [1/len(p_G.arg_matrices)] * len(p_G.arg_matrices)

    return jsonify(prior_distribution)

@app.route('/')
def index():
	return render_template('index.html')

if __name__ == "__main__":
    app.run()