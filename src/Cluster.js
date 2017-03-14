'use strict';

var Gaussian = require('./MultivariateGaussian');

class Cluster {
    constructor(weight, mu, sigma) {
        this.weight = weight;
        this.gaussian = new Gaussian({
            sigma: sigma,
            mu: mu
        });
    }

    probability(point) {
        return this.weight * this.gaussian.probability(point);
    }
}

module.exports = Cluster;