'use strict';

var Matrix = require('ml-matrix');

class MultivariateGaussian {
    constructor(parameters) {
        this.sigma = Matrix.checkMatrix(parameters.sigma);
        this.mu = Matrix.rowVector(parameters.mu);
        this.k = this.mu.columns;
        try {
            var det = this.sigma.det();
            this.sigmaInv = this.sigma.inverse();
            var sqrt2PI = Math.sqrt(Math.PI * 2);
            this.coeff = 1 / (Math.pow(sqrt2PI, this.k) * Math.sqrt(det));
            if ( !(isFinite(det) && det > 0 && isFinite(this.sigmaInv[0][0]))) {
                throw new Error();
            }
        } catch(e) {
            this.sigmaInv = Matrix.zeros(this.k, this.k);
            this.coeff = 0;
        }
    }

    probability(point) {
        point = Matrix.rowVector(point);
        var delta = point.sub(this.mu);

        var P = 0;
        for(var i = 0; i < this.k; i++) {
            var currentSigma = this.sigmaInv[i];
            var sum = 0;
            for(var j = 0; j < this.k; j++) {
                sum += currentSigma[j] * delta[0][j];
            }
            P += delta[0][i] * sum;
        }
        // Return: e^(-Π/2) / √|2.π.Σ|
        return this.coeff * Math.exp(P / -2);
    }
}

module.exports = MultivariateGaussian;