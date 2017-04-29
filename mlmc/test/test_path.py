import unittest
import numpy as np
import math

from mlmc.src.path import create_simple_path, create_layer_path
from mlmc.src.stock import ConstantVolatilityStock, VariableVolatilityStock
from mlmc.src.random_numbers import SampleCreator, IIDSampleCreator, CorrelatedSampleCreator

class ConstOneRng(SampleCreator):
    '''
    A rng that returns constants of value = 1
    '''
    def __init__(self, size=4):
        # need a no arg constructor
        super(ConstRng, self).__init__(size)
    
    def create_sample(self, n_samples=1, time_step=1, *args):
        x = np.array([1 for _ in xrange(n_samples)])
        return np.array([x for _ in xrange(self.size)])

class SimplePathTestCase(unittest.TestCase):
    '''
    To test create_simple_path()
    '''

    def test_stocks_input_unchanged(self):
        # create some stocks, record their current post_walk_price
        # run create_simple_path and confirm inputted stocks are not changed
        stocks = [ConstantVolatilityStock(i,1) for i in xrange(1,5)]
        current_price = [i for i in xrange(1,5)]
        _ = create_simple_path(stocks, risk_free=0.01, T=10, n_steps=252000)
        for j in xrange(len(stocks)):
            self.assertEqual(stocks[j].post_walk_price, current_price[j])

    def test_with_const_rng(self):
        base_vol, kappa, theta, gamma, r = 0.8, 0.5, 1.2, 1, 0.01
        stock_price = range(1,3)
        stocks = [VariableVolatilityStock(spot, base_vol, kappa, theta, gamma) for spot in stock_price]
        
        T, n_steps = 10, 252000
        dt = float(T) / n_steps
        var = base_vol**2
        dW, dZ = 1, 1 # because we are testing a rng that returns 1 only
        
        # first manually walk the stocks and store results in stock_price
        for i in xrange(n_steps):
            var += kappa * (theta - var) * dt + gamma * math.max(var,0)**0.5 * dZ
            for j in xrange(len(stock_price)):
                vol = math.max(var,0)**0.5
                stock_price[j] = stock_price[j] * math.exp((r - 0.5 * vol**2) * dt + vol * dW)
        
        # now use create_simple_path() to walk
        post_walk_price = create_simple_path(stocks, r, T, n_steps, ConstOneRng)

        for j in xrange(len(stock_price)):
            self.assertAlmostEqual(stock_price[j], post_walk_price[j])

class LayerPathTestCase(unittest.TestCase):
    '''
    To test create_layer_path()
    '''
    def test_with_const_rng(self):
        base_vol, kappa, theta, gamma, r = 0.8, 0.5, 1.2, 1, 0.01
        stock_price = range(1,3)
        stocks = [VariableVolatilityStock(spot, base_vol, kappa, theta, gamma) for spot in stock_price]
        
        T, n_steps = 10, 252000
        dt = float(T) / n_steps
        var = base_vol**2
        dW, dZ = 1, 1 # because we are testing a rng that returns 1 only

if __name__ == '__main__':
    unittest.main()