'use strict';

const ExpectationMaximization = require('..');
const MG = require('../src/MultivariateGaussian');
const Matrix = require('ml-matrix');

describe('ml-expectation-maximization test', function () {

    it('basic test', function () {
        var points = [
            [    241,    253  ],
            [    1240,    214  ],
        ];
        var groups = new ExpectationMaximization();
        groups.train(points);

        groups.clusters[0].weight.should.be.approximately(.5, 1e-3);
        groups.clusters[1].weight.should.be.approximately(.5, 1e-3);
    });
    
    it('Example with 500 points', function () {
        var size = 500;

        var points = [];
        var groups = [{
            weight: .3,
            mu: Matrix.rowVector([5, 5])
        }, {
            weight: .7,
            mu: Matrix.rowVector([0, 1])
        }];


        var sub = Matrix.rowVector([.5, .5]);
        var counter0 = 0, counter1 = 0;
        for (var i = 0; i < size; i++) {
            var group = Math.random() < groups[0].weight ? 0 : 1;
            if(group === 0) {
                counter0++;
            } else {
                counter1++;
            }
            var mu = groups[group].mu;
            var r = mu.clone();
            var N = 10;
            for (var k = 0; k < N; k++) {
                r.add(Matrix.rand(1, 2).sub(sub));
            }
            points.push(r[0]);
        }

        var em = new ExpectationMaximization();
        em.train(points);

        em.clusters.sort(function (a, b) {
            return a.weight - b.weight;
        });

        em.clusters[0].weight.should.be.approximately(groups[0].weight, 0.1);
        em.clusters[1].weight.should.be.approximately(groups[1].weight, 0.1);
        Matrix.sub(em.clusters[0].gaussian.mu, groups[0].mu).abs().sum().should.be.approximately(0, 0.3);
        Matrix.sub(em.clusters[1].gaussian.mu, groups[1].mu).abs().sum().should.be.approximately(0, 0.3);
        /*assert.ok(isNear(result[0].weight, groups[0].weight));
        assert.ok(isNear(result[0].mu, groups[0].mu));
        assert.ok(isNear(result[1].weight, groups[1].weight));
        assert.ok(isNear(result[1].mu, groups[1].mu));
        assert.end();*/
    })
});

describe("Multivariate Gaussian Test", function () {
    it('basic test', function () {
        var params = {
            sigma: [
                [3.10853404, 0.57142415, 0.03101091],
                [0.9549752, 3.89398613, 0.88597582],
                [0.87729471, 0.82066072, 3.67113053]
            ],
            mu: [
                [0.73516845, 0.27666293, 0.65376305]
            ]
        };
        var x = [0.7,  0.2,  0.6];
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
            mu : [
                [0, 0]
            ]
        };

        var gaussian = new MG(params);
        var res = gaussian.probability([0, 0]);
        res.should.be.approximately(0, 1e-3);
    });

    it('works when sigma^-1 is large', function() {
        // Test when sigma^-1 is very large
        var params = {
            sigma: [
                [72.24556085089353, -84.99477747163944],
                [-84.99477747163944, 99.99385584898758]
            ],
            mu: [
                [0, 0]
            ]
        };

        var gaussian = new MG(params);
        var res = gaussian.probability([0, 0]);
        res.should.be.approximately(0, 1e-3);
    });
});