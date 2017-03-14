'use strict';

var Matrix = require('ml-matrix');

class MultivariateGaussian {
    constructor(parameters) {
        this.sigma = Matrix.checkMatrix(parameters.sigma);
        this.mu = Matrix.checkMatrix(parameters.mu);
        this.k = this.mu.columns;
        try {
            // TODO: try with this.sigma.det()
            var det = Matrix.DC.LuDecomposition(this.sigma).determinant;
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
        var delta = Matrix.rowVector(point).sub(this.mu);

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