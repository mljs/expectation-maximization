'use strict';

const Cluster = require('./Cluster');
const Matrix = require('ml-matrix');
const MT = require('ml-xsadd');

const defaultOptions = {
    epsilon: 2e-16,
    clusters: 2,
    maxIterations: 1000,
    seed: 42
};

class ExpectationMaximization{
    constructor(options) {
        options = Object.assign({}, defaultOptions, options);
        this.epsilon = options.epsilon;
        this.numClusters = options.clusters;
        this.maxIters = options.maxIterations;
        this.seed = options.seed;
    }

    train(features) {
        features = Matrix.checkMatrix(features);
        var estimations = Matrix.rand(this.numClusters, features.rows, new MT(this.seed).random);
        for(var i = 0; i < this.maxIters; ++i) {
            var clusters = this.maximization(features, estimations);
            var oldEstimations = estimations.clone();
            estimations = this.expectation(features, clusters);
            var delta = Matrix.sub(estimations, oldEstimations).abs().max();
            if(delta <= this.epsilon) {
                break;
            }
        }

        this.clusters = clusters;
    }

    predict(features) {
        var predictions = new Array(features.rows);
        for(var i = 0; i < features.rows; ++i) {
            var max = 0, maxIndex = 0;
            for(var j = 0; j < this.clusters.length; ++j) {
                var currentProb = this.clusters[j].probability(features[i]);
                if(currentProb > max) {
                    max = currentProb;
                    maxIndex = j;
                }
            }
            predictions[i] = maxIndex;
        }

        return predictions;
    }

    maximization(features, estimations) {
        var len = estimations.rows;
        var dim = features.columns;
        var res = new Array(len);
        var sum = estimations.sum();
        var sumByRow = estimations.sum('row').to1DArray();
        var featuresT = features.transpose();
        for(var g = 0; g < len; g++) {
            var currentEstimation = estimations.getRowVector(g);
            var estimationSum = sumByRow[g];
            if (estimationSum < this.epsilon) {
                currentEstimation.fill(this.epsilon);
                estimationSum = currentEstimation.length * this.epsilon;
            }
            // Compute the weight
            var weight = estimationSum / sum;
            // Compute the mean
            var mu = Matrix.div(featuresT, estimationSum);
            for(var m = 0; m < mu.length; m++) {
                mu[m] = mu.getRowVector(m).mul(currentEstimation).sum();
            }
            //mu = mu.sum('row').transpose();
            mu = Matrix.rowVector(mu);
            // Compute the covariance
            var sigma = Matrix.diag(new Matrix(1, dim).fill(this.epsilon)[0]);
            //var sigma = n.diag(n.rep([dim], n.epsilon));
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
            res[g] = new Cluster(weight, mu, sigma);
        }
        return res;
    }

    expectation(features, clusters) {
        var res = new Array(features.rows);// new Array(points.length);

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
            // Convert to probabilities by dividing by the sum
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
