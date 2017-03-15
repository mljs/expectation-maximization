# expectation-maximization

  [![NPM version][npm-image]][npm-url]
  [![build status][travis-image]][travis-url]
  [![David deps][david-image]][david-url]
  [![npm download][download-image]][download-url]

Gaussian Mixture Models using Expectation Maximization algorithm transcribed from Expectation Maximization [repository](https://github.com/lovasoa/expectation-maximization)
of Ophir LOJKINE, also using his multivariate gaussian [link](https://github.com/lovasoa/multivariate-gaussian).

## Installation

`$ npm install ml-expectation-maximization`

## [API Documentation](https://mljs.github.io/expectation-maximization/)

## Example

```js
const EM = require('ml-expectation-maximization');
em.train(data); // data is a Training matrix
em.predict(toPredict); // data matrix to predict

// save your model
var model = em.toJSON();

// load your model
var newEM = EM.load(model);
```

Or test it in [Runkit](https://runkit.com/npm/ml-expectation-maximization)

## License

[MIT](./LICENSE)

[npm-image]: https://img.shields.io/npm/v/ml-expectation-maximization.svg?style=flat-square
[npm-url]: https://npmjs.org/package/ml-expectation-maximization
[travis-image]: https://img.shields.io/travis/mljs/ml-expectation-maximization/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/mljs/ml-expectation-maximization
[david-image]: https://img.shields.io/david/mljs/ml-expectation-maximization.svg?style=flat-square
[david-url]: https://david-dm.org/mljs/ml-expectation-maximization
[download-image]: https://img.shields.io/npm/dm/ml-expectation-maximization.svg?style=flat-square
[download-url]: https://npmjs.org/package/ml-expectation-maximization
