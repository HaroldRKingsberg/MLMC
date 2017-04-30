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
        return (float(dZ)/time_step for dZ in vol_steps)

    def walk_price(self, risk_free, time_step, price_steps, vol_steps):
        vols = self.find_volatilities(time_step, vol_steps)

        change = sum(
            time_step*risk_free + sigma*dW
            for sigma, dW in zip(vols, price_steps)
        )
        self.post_walk_price = self.spot * change
        return self.post_walk_price


class MockSampleCreator(SampleCreator):

    def __init__(self, samples):
        super(MockSampleCreator, self).__init__(len(samples))

        self.samples = samples

    def create_sample(self, n_samples=1, time_step=1, *args):
        return np.array([
            (subsample * time_step)[:n_samples]
            for subsample in self.samples
        ])


class ConstRng(SampleCreator):
    '''
    A rng that returns constants of value = 0.01
    '''
    def __init__(self):
        super(ConstRng, self).__init__(4)
    
    def create_sample(self, n_samples=1, time_step=1, *args):
        x = np.array([0.01 for _ in xrange(n_samples)])
        return np.array([x for _ in xrange(self.size)])


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
        expecteds = [42.55, 420.1]

        for observed, expected in zip(paths, expecteds):
            self.assertAlmostEqual(observed, expected)

"""
class LayerPathTestCase(unittest.TestCase):
    '''
    To test create_layer_path()
    '''
    def test_with_const_rng(self):
        base_vol, kappa, theta, gamma, r = 0.8, 0.5, 1.2, 1, 0.01
        stock_price = range(1,3)
        stocks = [VariableVolatilityStock(spot, base_vol, kappa, theta, gamma) for spot in stock_price]
        
        # n_steps := num steps at coarse level
        T, n_steps = 10, 999
        dt = float(T) / n_steps
        var = base_vol**2
        dW, dZ = 0.01, 0.01 # because we are testing a rng that returns 0.01
        K = 2

        # manually walk at the fine level
        # n_steps * K = num steps at fine level
        stock_price_fine = list(stock_price)
        for i in xrange(n_steps * K):
            var += kappa * (theta - var) * dt + gamma * max(var,0)**0.5 * dZ
            for j in xrange(len(stock_price_fine)):
                vol = max(var,0)**0.5
                stock_price_fine[j] = stock_price_fine[j] * math.exp((r - 0.5 * vol**2) * dt + vol * dW)

        # manually walk at the coarse level
        # n_steps on one path at coarse level but each step size = K * dW
        stock_price_coarse = list(stock_price)
        var = base_vol**2
        for i in xrange(n_steps):
            var += kappa * (theta - var) * dt + gamma * max(var,0)**0.5 * (dZ * K)
            for j in xrange(len(stock_price_coarse)):
                vol = max(var,0)**0.5
                stock_price_coarse[j] = stock_price_coarse[j] * math.exp((r - 0.5 * vol**2) * dt + vol * (dW * K))

        manual_walk = zip(stock_price_coarse, stock_price_fine)

        # now using layer path function
        post_walk_price = create_layer_path(stocks, r, T, n_steps, ConstRng)

        for j in xrange(len(post_walk_price)):
            tup_manual = manual_walk[j]
            tup_func = post_walk_price[j]
            for k in xrange(1):
                self.assertAlmostEqual(tup_manual[k], tup_func[k]) 
"""
if __name__ == '__main__':
    unittest.main()
