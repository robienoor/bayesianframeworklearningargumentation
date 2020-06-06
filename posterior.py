import numpy as np

def calculatePosteriorDistribution(prior_distribution, liklihood_distribution):

    product = prior_distribution * liklihood_distribution
    normalising_constant = np.sum(product)

    posterior_distribution = product / normalising_constant
 
    return posterior_distribution   


class posterior():

    def __init__(self, prior_distribution, liklihood_distribution):

        self.prior_distribution = prior_distribution
        self.liklihood_distribution = liklihood_distribution

    def buildPosteriorDistribution(self):
        self.posterior_distribution = calculatePosteriorDistribution(self.prior_distribution, self.liklihood_distribution)

        return self.posterior_distribution

