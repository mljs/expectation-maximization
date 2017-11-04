import {ExpectationMaximization} from '../ExpectationMaximization';
import {MultivariateGaussian as MG} from '../MultivariateGaussian';
import Matrix from 'ml-matrix';

describe('expectation-maximization test', () => {

    function getModel(size = 500) {
        var points = [];
        var groups = [{
            weight: .3,
            mu: Matrix.rowVector([5, 5])
        }, {
            weight: .7,
            mu: Matrix.rowVector([0, 1])
        }];


        var sub = Matrix.rowVector([.5, .5]);
        for (var i = 0; i < size; i++) {
            var group = Math.random() < groups[0].weight ? 0 : 1;
            var mu = groups[group].mu;
            var r = mu.clone();
            var N = 10;
            for (var k = 0; k < N; k++) {
                r.add(Matrix.rand(1, 2).sub(sub));
            }
            points.push(r[0]);
        }

        var em = new ExpectationMaximization({
            numClusters: groups.length,
            seed: 42
        });
        em.train(points);

        return {
            model: em,
            groups: groups
        };
    }

    test('Example with 500 points', function () {
        var result = getModel();
        var em = result.model;
        var groups = result.groups;

        var clusterData = em.getClusterData();

        clusterData.sort(function (a, b) {
            return a.weight - b.weight;
        });

        for (var i = 0; i < groups.length; ++i) {
            expect(clusterData[i].weight).toBeCloseTo(groups[i].weight, 1);
            expect(Matrix.sub(clusterData[i].mean, groups[i].mu).abs().sum()).toBeCloseTo(0, 0);
            expect(clusterData[i]).toHaveProperty('mean');
            expect(clusterData[i]).toHaveProperty('covariance');
            expect(clusterData[i]).toHaveProperty('prediction');
            expect(clusterData[i]).toHaveProperty('weight');
        }

        // this is made because initialization of the clusters
        em.clusters.sort(function (a, b) {
            return a.weight - b.weight;
        });

        var predictions = em.predict([[4, 4], [0, 0]]);

        expect(predictions[0]).toBe(0);
        expect(predictions[1]).toBe(1);
    });

    test('save and load', () => {
        var em = getModel().model;

        // this is made because initialization of the clusters
        em.clusters.sort(function (a, b) {
            return a.weight - b.weight;
        });

        var newModel = ExpectationMaximization.load(JSON.parse(JSON.stringify(em)));

        var predictions = newModel.predict([[4, 4], [0, 0]]);
        expect(predictions[0]).toBe(0);
        expect(predictions[1]).toBe(1);
    });

    test('case with singular matrix', function () {
        var points = [
            [241, 253],
            [1240, 214],
        ];
        var em = new ExpectationMaximization({
            seed: 42
        });
        em.train(points);

        expect(em.clusters[0].weight).toBeCloseTo(.5, 2);
        expect(em.clusters[1].weight).toBeCloseTo(.5, 2);
    });
});

describe('Multivariate Gaussian Test', () => {
    test('basic test', () => {
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
        var x = [0.7, 0.2, 0.6];
        var gaussian = new MG(params);
        var prob = gaussian.probability(x);

        expect(prob).toBeCloseTo(0.010375829337330682, 2);
    });

    test('handles singular matrices', () => {
        var params = {
            sigma: [
                [1, 1],
                [1, 1]
            ],
            mu: [
                [0, 0]
            ]
        };

        var gaussian = new MG(params);
        var res = gaussian.probability([0, 0]);
        expect(res).toBeCloseTo(0, 2);
    });

    test('works when sigma^-1 is large', () => {
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
        expect(res).toBeCloseTo(0, 2);
    });
});
