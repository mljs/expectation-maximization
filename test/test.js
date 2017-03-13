'use strict';

const ExpectationMaximization = require('..');
const MG = require('../src/MultivariateGaussian');
const Matrix = require('ml-matrix');

describe('ml-expectation-maximization test', function () {

    it('Something to test', function () {
        var a = new Matrix([
            [3, 4],
            [5, 6]
        ]);

        var b = a.clone();

        var c = Matrix.add(a, b);
    });
});

describe("Multivariate Gaussian Test", function () {
    it('basic test', function () {
        var params = {
            sigma : [[ 3.10853404,  0.57142415,  0.03101091],
                [ 0.9549752 ,  3.89398613,  0.88597582],
                [ 0.87729471,  0.82066072,  3.67113053]],

            mu :     [ 0.73516845,  0.27666293,  0.65376305]
        }
        var x = [ 0.7,  0.2,  0.6];
        var gaussian = new MG(params);
        var prob = gaussian.probability(x);

        prob.should.be.approximately(0.010375829337330682, 1e-3);
    });

    it('handles singular matrices', function() {
        var params = {
            sigma : [
                [1, 1],
                [1, 1]
            ],
            mu : [0, 0]
        };

        var gaussian = new MG(params);
        var res = gaussian.probability([0, 0]);
        res.should.be.approximately(0, 1e-3);
    });

    it('works when sigma^-1 is large', function() {
        // Test when sigma^-1 is very large
        var params = {
            // n*n covariance matrix
            sigma : [
                [ 72.24556085089353, -84.99477747163944],
                [-84.99477747163944, 99.99385584898758]
            ],
            mu : [0, 0]
        };

        var gaussian = new MG(params);
        var res = gaussian.probability([0, 0]); // Returns the value of the density function at (0,0)
        res.should.be.approximately(0, 1e-3);
    });
});