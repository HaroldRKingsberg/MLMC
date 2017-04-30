import functools
import unittest
import numpy as np
import math

from mlmc.path import create_simple_path, create_layer_path
from mlmc.stock import ConstantVolatilityStock, VariableVolatilityStock, Stock
from mlmc.random_numbers import SampleCreator, IIDSampleCreator, CorrelatedSampleCreator


class MockStock(Stock):

    '''
    Garbage class used exclusively for the purposes of testing
    that paths work. All variables used, but in a rudimentary
    fashion that allows for pen-and-paper testing
    '''

    def find_volatilities(self, time_step, vol_steps):
        for dZ in vol_steps:
            yield float(dZ) / time_step

    def walk_price(self, risk_free, time_step, price_steps, vol_steps):
        price = self.spot
        vols = self.find_volatilities(time_step, vol_steps)

        for sigma, dW in zip(vols, price_steps):
            price *= (time_step*risk_free + sigma*dW)

        self.post_walk_price = price
        return price


class MockSampleCreator(SampleCreator):

    def __init__(self, samples):
        super(MockSampleCreator, self).__init__(len(samples))

        self.samples = samples

    def create_sample(self, n_samples=1, time_step=1, *args):
        return np.array([
            (subsample * time_step)[:n_samples]
            for subsample in self.samples
        ])


class SimplePathTestCase(unittest.TestCase):

    def test_all_stocks_walked(self):
        '''
        The stock objects used as inputs should not themselves be walked
        '''
        stocks = [ConstantVolatilityStock(i, 1) for i in xrange(1, 5)]
        paths = create_simple_path(stocks, risk_free=0.01, T=10, n_steps=5)
        self.assertEqual(len(paths), len(stocks))

    def test_stocks_input_unchanged(self):
        '''
        The stock objects used as inputs should not themselves be walked
        '''
        stocks = [ConstantVolatilityStock(i, 1) for i in xrange(1, 5)]
        create_simple_path(stocks, risk_free=0.01, T=10, n_steps=5)

        for spot, stock in enumerate(stocks, start=1):
            self.assertEqual(stock.post_walk_price, spot)

    def test_path_creation_will_walk_all_steps(self):
        spots = map(float, xrange(1, 3))
        r = 0.01
        T = 5
        n_steps = 2
        stocks = [MockStock(s) for s in spots]

        samples = np.array([
            np.array([5, 6, 7]),
            np.array([1, 2, 3]),
            np.array([9, 8, 7]),
            np.array([4, 6, 2]),
        ])

        rng_creator = functools.partial(MockSampleCreator, samples)
        paths = create_simple_path(stocks, r, T, n_steps, rng_creator)
        expecteds = [376.063125, 21610.50125]

        for observed, expected in zip(paths, expecteds):
            self.assertAlmostEqual(observed, expected)


class LayerPathTestCase(unittest.TestCase):
    '''
    To test create_layer_path()
    '''
    def test_with_const_rng(self):
        spots = map(float, xrange(1, 3))
        r = 0.01
        T = 5
        n_steps = 1
        stocks = [MockStock(s) for s in spots]

        samples = np.array([
            np.array([5, 6, 7]),
            np.array([1, 2, 3]),
            np.array([9, 8, 7]),
            np.array([4, 6, 2]),
        ])

        rng_creator = functools.partial(MockSampleCreator, samples)
        paths = create_layer_path(stocks, r, T, n_steps, rng_creator)
        expecteds = [
            (41.3, 376.063125), 
            (425.1, 21610.50125),
        ]

        for (coarse_o, fine_o), (coarse_e, fine_e) in zip(paths, expecteds):
            self.assertAlmostEqual(coarse_o, coarse_e)
            self.assertAlmostEqual(fine_o, fine_e)


if __name__ == '__main__':
    unittest.main()
