import {Cluster} from './Cluster';
import Matrix from 'ml-matrix';
import expectationMaximization from 'expectation-maximization';

const defaultOptions = {
    epsilon: 2e-16,
    numClusters: 2,
    seed: undefined
};

/**
 * @class ExpectationMaximization
 */
export class ExpectationMaximization {

    /**
     * Constructor for Expectation Maximization
     * @param {object} options
     * @param {number} [options.epsilon=2e-16] : Convergence threshold for final solution.
     * @param {number} [options.numClusters=2] : Number of clusters to find.
     * @param {number} [options.seed=undefined] : Seed for the random generator
     */
    constructor(options) {
        options = Object.assign({}, defaultOptions, options);
        this.epsilon = options.epsilon;
        this.numClusters = options.numClusters;
        this.maxIterations = options.maxIterations;
        this.seed = options.seed;
        if (options.model === 'em-gmm') {
            this.clusters = new Array(this.numClusters);
            for (var i = 0; i < this.numClusters; ++i) {
                this.clusters[i] = Cluster.load(options.clusters[i]);
            }
        }
    }

    /**
     * Train the current model with the given cases
     * @param {Matrix|Array} features
     */
    train(features) {
        var groups = expectationMaximization(features, this.numClusters);
        var clusters = new Array(groups.length);
        for (var i = 0; i < groups.length; ++i) {
            var currentGroup = groups[i];
            clusters[i] = new Cluster({
                weight: currentGroup.weight,
                mu: [currentGroup.mu],
                sigma: currentGroup.sigma
            });
        }

        this.clusters = clusters;
    }

    /**
     * Predict the output of each element of the matrix
     * @param {Matrix|Array} features
     * @return {Array} : predictions
     */
    predict(features) {
        features = Matrix.checkMatrix(features);
        var predictions = new Array(features.rows);
        for (var i = 0; i < features.rows; ++i) {
            var max = 0;
            var maxIndex = 0;
            for (var j = 0; j < this.clusters.length; ++j) {
                var currentProb = this.clusters[j].probability(features[i]);
                if (currentProb > max) {
                    max = currentProb;
                    maxIndex = j;
                }
            }
            predictions[i] = maxIndex;
        }

        return predictions;
    }

    /**
     * Returns an array of objects containing all the information related to each cluster (Note: this method is only valid after running the train method).
     * Each element of the array contains
     *
     *  * weight: Weight of the current cluster.
     *  * mean: Current mean of the cluster.
     *  * covariance: Covariance matrix of the cluster.
     *  * prediction: prediction label associated with the cluster.
     *
     * @return {Array}
     */
    getClusterData() {
        var clusterData = new Array(this.numClusters);
        for (var i = 0; i < this.numClusters; ++i) {
            var currentCluster = this.clusters[i];
            clusterData[i] = {
                weight: currentCluster.weight,
                mean: currentCluster.gaussian.mu,
                covariance: currentCluster.gaussian.sigma,
                prediction: i
            };
        }

        return clusterData;
    }

    /**
     * Save the current model to JSON format
     * @return {object} model
     */
    toJSON() {
        return {
            model: 'em-gmm',
            clusters: this.clusters,
            epsilon: this.epsilon,
            numClusters: this.numClusters,
            maxIterations: this.maxIterations,
            seed: this.seed
        };
    }

    /**
     * Load a Expectation-Maximization with the given model.
     * @param {object} model
     * @return {ExpectationMaximization}
     */
    static load(model) {
        if (model.model !== 'em-gmm') {
            throw new RangeError('the current model is invalid!');
        }

        return new ExpectationMaximization(model);
    }
}
