'use strict';

const Cluster = require('./Cluster');
const Matrix = require('ml-matrix');

const defaultOptions = {
    epsilon: 2e-16,
    clusters: 2,
    maxIterations: 1000
};

class ExpectationMaximization{
    constructor(options) {
        options = Object.assign({}, defaultOptions, options);
        this.epsilon = options.epsilon;
        this.clusters = options.clusters;
        this.maxIters = options.maxIterations;
    }

    train(features) {
        var estimations = Matrix.rand(this.clusters, features.rows);
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

    maximization(features, estimations) {
        var len = estimations.rows;
        var dim = estimations.columns;
        var res = new Array(len);
        var sum = estimations.sum();
        var sumByRow = estimations.sum('row');
        var featuresT = features.transposeView();
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
            // Compute the covariance
            var sigma = new Matrix(1, dim).fill(this.epsilon);
            //var sigma = n.diag(n.rep([dim], n.epsilon));
            for (var i = 0; i < len; i++) {
                var point = features.getRowVector(i);
                var diff = Matrix.sub(point, mu);
                var coeff = currentEstimation[i] / estimationSum;
                for (var a = 0; a < diff.length; a++) {
                    for (var b = 0; b <= a; b++) {
                        var tmp = coeff * diff[a] * diff[b];
                        sigma[a][b] += tmp;
                        if (b !== a) sigma[b][a] += tmp;
                    }
                }
            }
            res[g] = new Cluster(weight, mu, sigma);
        }
        return res;
    }

    expectation(features, clusters) {
        var res = new Matrix(1, features.columns);// new Array(points.length);

        for (var p = 0; p < features.rows; p++) {
            var point = features[p];
            var line = new Array(this.clusters);
            var sum = 0;
            // Compute the raw density values
            for (var g = 0; g < this.clusters; g++) {
                var prob = clusters[g].probability(point);
                line[g] = prob;
                sum += prob;
            }
            // Convert to probabilities by dividing by the sum
            if (sum > 0) {
                for (g = 0; g < this.clusters; g++) {
                    line[g] /= sum;
                }
            } else {
                for (g = 0; g < this.clusters; g++) {
                    line[g] = 1 / this.clusters;
                }
            }
            res[p] = line;
        }

        return res.transpose();
    }
}

module.exports = ExpectationMaximization;
