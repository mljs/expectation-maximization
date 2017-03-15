'use strict';

var Gaussian = require('./MultivariateGaussian');

class Cluster {

    /**
     * @private
     *
     * Constructor for a cluster of the EM algorithm.
     *
     * @param {object} parameters
     * @param {number} [parameters.weight] : weight of the current cluster (must be between 0-1)
     * @param {Matrix|Array} [parameters.mu] : Mean of the cluster.
     * @param {Matrix|Array} [parameters.sigma] : Covariance matrix of the cluster.
     * @param {boolean} load
     */
    constructor(parameters, load) {
        this.weight = parameters.weight;

        if (load) {
            this.gaussian = Gaussian.load(parameters.gaussian);
        } else {
            this.gaussian = new Gaussian({
                sigma: parameters.sigma,
                mu: parameters.mu
            });
        }
    }

    /**
     * @private
     * Calculates the probability of a given point of belonging to the cluster
     * @param {Array} point
     * @return {number} : probability
     */
    probability(point) {
        return this.weight * this.gaussian.probability(point);
    }

    /**
     * @private
     * Save the current cluster model.
     * @return {{weight: *, gaussian: (MultivariateGaussian|*|Gaussian)}}
     */
    toJSON() {
        return {
            weight: this.weight,
            gaussian: this.gaussian
        };
    }

    /**
     * @private
     * Load a new cluster with the given model.
     * @param {object} model
     * @return {Cluster}
     */
    static load(model) {
        return new Cluster(model, true);
    }
}

module.exports = Cluster;
