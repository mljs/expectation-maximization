'use strict';

const Cluster = require('./Cluster');
const Matrix = require('ml-matrix');
const MT = require('ml-xsadd');

const defaultOptions = {
    epsilon: 2e-16,
    numClusters: 2,
    maxIterations: 1000,
    seed: undefined
};

/**
 * @class ExpectationMaximization
 */
class ExpectationMaximization {

    /**
     * Constructor for Expectation Maximization
     * @param {object} options
     * @param {number} [options.epsilon=2e-16] : Convergence threshold for final solution.
     * @param {number} [options.numClusters=2] : Number of clusters to find.
     * @param {number} [options.maxIterations=1000] : Maximum number of iterations of the algorithm.
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
        features = Matrix.checkMatrix(features);
        var featuresT = features.transpose();
        var estimations = Matrix.rand(this.numClusters, features.rows, new MT(this.seed).random);
        for (var i = 0; i < this.maxIterations; ++i) {
            var clusters = this.maximization(features, featuresT, estimations);
            var oldEstimations = estimations.clone();
            estimations = this.expectation(features, clusters);
            var delta = Matrix.sub(estimations, oldEstimations).abs().max();
            if (delta <= this.epsilon) {
                break;
            }
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
            var max = 0, maxIndex = 0;
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

    /**
     * @private
     *
     * Maximization step of the algorithm
     *
     * @param {Matrix} features : training set
     * @param {Matrix} featuresT : training set transposed
     * @param {Matrix} estimations : estimations of the expectation step or initial random guess.
     * @return {Array} : Array of gaussian multivariate clusters.
     */
    maximization(features, featuresT, estimations) {
        var len = estimations.rows;
        var dim = features.columns;
        var res = new Array(len);
        var sum = estimations.sum();
        var sumByRow = estimations.sum('row');
        // var featuresT = features.transpose();
        for (var g = 0; g < len; g++) {
            var currentEstimation = estimations.getRowVector(g);
            var estimationSum = sumByRow[g][0];
            if (estimationSum < this.epsilon) {
                currentEstimation.fill(this.epsilon);
                estimationSum = currentEstimation.columns * this.epsilon;
            }
            // Compute weight
            var weight = estimationSum / sum;
            // Compute mean
            var mu = Matrix.div(featuresT, estimationSum);
            for (var m = 0; m < mu.length; m++) {
                mu[m] = mu.getRowVector(m).mul(currentEstimation).sum();
            }

            mu = Matrix.rowVector(mu);
            // Compute the covariance
            var sigma = Matrix.diag(new Matrix(1, dim).fill(this.epsilon)[0]);
            for (var i = 0; i < features.rows; i++) {
                var point = features.getRowVector(i);
                var diff = Matrix.sub(point, mu);
                var coeff = currentEstimation[0][i] / estimationSum;
                for (var a = 0; a < diff.columns; a++) {
                    for (var b = 0; b <= a; b++) {
                        var tmp = coeff * diff[0][a] * diff[0][b];
                        sigma[a][b] += tmp;
                        if (b !== a) {
                            sigma[b][a] += tmp;
                        }
                    }
                }
            }
            res[g] = new Cluster({
                weight: weight,
                mu: mu,
                sigma: sigma
            });
        }
        return res;
    }

    /**
     * @private
     *
     * Expectation step of the algorithm.
     *
     * @param {Matrix} features : Training set.
     * @param {Array} clusters : Array of multivariate gaussian clusters.
     * @return {Matrix} : New estimations with the given clusters.
     */
    expectation(features, clusters) {
        var res = new Array(features.rows);

        for (var p = 0; p < features.rows; p++) {
            var point = features[p];
            var line = new Array(this.numClusters);
            var sum = 0;
            // Compute the raw density values
            for (var g = 0; g < this.numClusters; g++) {
                var prob = clusters[g].probability(point);
                line[g] = prob;
                sum += prob;
            }

            if (sum > 0) {
                for (g = 0; g < this.numClusters; g++) {
                    line[g] /= sum;
                }
            } else {
                for (g = 0; g < this.numClusters; g++) {
                    line[g] = 1 / this.numClusters;
                }
            }
            res[p] = line;
        }

        return new Matrix(res).transpose();
    }
}

module.exports = ExpectationMaximization;
