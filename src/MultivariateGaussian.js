import {Matrix, inverse} from 'ml-matrix';

export class MultivariateGaussian {
    /**
     * @private
     *
     * Constructor of the multivariate gaussian distribution
     *
     * @param {object} parameters
     * @param {Matrix|Array} [parameters.mu] - Mean Matrix
     * @param {Matrix|Array} [parameters.sigma] - Covariance Matrix
     * @param {boolean} load
     */
    constructor(parameters, load) {
        if (load) {
            this.k = parameters.k;
            this.mu = parameters.mu;
            this.sigma = parameters.sigma;
            this.sigmaInv = parameters.sigmaInv;
            this.coeff = parameters.coeff;
        } else {
            this.sigma = Matrix.checkMatrix(parameters.sigma);
            this.mu = Matrix.checkMatrix(parameters.mu);
            this.k = this.mu.columns;
            try {
                var det = this.sigma.det();
                this.sigmaInv = inverse(this.sigma);
                var sqrt2PI = Math.sqrt(Math.PI * 2);
                this.coeff = 1 / (Math.pow(sqrt2PI, this.k) * Math.sqrt(det));
                if (!(Number.isFinite(det) && det > 0 && Number.isFinite(this.sigmaInv[0][0]))) {
                    throw new Error('Infinte determinant for Multivariate gaussian');
                }
            } catch (e) {
                this.sigmaInv = Matrix.zeros(this.k, this.k);
                this.coeff = 0;
            }
        }
    }

    /**
     * @private
     *
     * Calculates the probability density function(also knows as pdf) at the given point.
     *
     * @param {Array} point
     * @return {number}
     */
    probability(point) {
        var delta = Matrix.rowVector(point).sub(this.mu);

        var P = 0;
        for (var i = 0; i < this.k; i++) {
            var currentSigma = this.sigmaInv[i];
            var sum = 0;
            for (var j = 0; j < this.k; j++) {
                sum += currentSigma[j] * delta[0][j];
            }
            P += delta[0][i] * sum;
        }

        return this.coeff * Math.exp(P / -2);
    }


    /**
     * @private
     *
     * Save the current MultivariateGaussian model.
     *
     * @return {object} : JSON object.
     */
    toJSON() {
        return {
            k: this.k,
            mu: this.mu,
            sigma: this.sigma,
            sigmaInv: this.sigmaInv,
            coeff: this.coeff
        };
    }


    /**
     * @private
     *
     * Load a new MultivariateGaussian with the given model.
     *
     * @param {object} model
     * @return {MultivariateGaussian}
     */
    static load(model) {
        return new MultivariateGaussian(model, true);
    }
}
