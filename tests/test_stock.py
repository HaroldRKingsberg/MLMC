import itertools
import math
import unittest

from mlmc.stock import (Stock,
                            ConstantVolatilityStock,
                            VariableVolatilityStock)

class StockTestCase(unittest.TestCase):

    class TestStock(Stock):
        def __init__(self, spot):
            super(StockTestCase.TestStock, self).__init__(spot)
            self.vol_calls = []

        def find_volatilities(self, time_step, vol_steps):
            self.vol_calls.append(vol_steps)
            return itertools.repeat(1)

    def test_cannot_instantiated_without_find_volatilites(self):
        with self.assertRaises(TypeError):
            Stock(10)

    def test_find_volatilities_implementation_allows_instantiation(self):
        try:
            self.TestStock(10)
        except TypeError:
            self.fail()

    def test_walking_price_changes_post_walk_price(self):
        spot = 10
        risk_free = 0.025
        time_step = 1
        price_steps = [0.075]
        vol_steps = [0.25]
        sut = self.TestStock(spot)

        sut.walk_price(risk_free,
                       time_step,
                       price_steps,
                       vol_steps)
        self.assertEqual(sut.spot, spot)
        self.assertNotEqual(sut.spot, sut.post_walk_price)

    def test_walking_price_takes_deterministic_step(self):
        spot = 10
        risk_free = 0.025
        time_step = 0.0025
        price_steps = [0]
        vol_steps = [0]
        sut = self.TestStock(spot)

        sut.walk_price(risk_free,
                       time_step,
                       price_steps,
                       vol_steps)
        self.assertAlmostEqual(sut.post_walk_price,
                               (1 + risk_free*time_step) * spot)

    def test_walking_price_takes_multiple_deterministic_steps(self):
        spot = 10
        risk_free = 0.025
        time_step = 0.0025
        price_steps = [0, 0]
        vol_steps = [0, 0]
        sut = self.TestStock(spot)

        sut.walk_price(risk_free,
                       time_step,
                       price_steps,
                       vol_steps)
        self.assertAlmostEqual(sut.post_walk_price,
                               ((1 + risk_free*time_step)**2) * spot)


    def test_walking_price_deterministic_approaches_exponential(self):
        spot = 10
        risk_free = 0.1
        N = int(1e7)
        t = 1.0
        price_steps = [0 for _ in range(N)]
        time_step = t/N

        sut = self.TestStock(spot)
        expected = spot * math.exp(risk_free*t)
        sut.walk_price(risk_free,
                       time_step,
                       price_steps)
        self.assertAlmostEqual(sut.post_walk_price,
                               expected,
                               5)

    def test_walking_price_allows_random_walk(self):
        spot = 10
        risk_free = 0
        time_step = 1
        price_steps = [0.25, 0.5]
        vol_steps = [0, 0]
        sut = self.TestStock(spot)
        expected = 12.50 + 6.25

        sut.walk_price(risk_free,
                       time_step,
                       price_steps,
                       vol_steps)
        self.assertAlmostEqual(sut.post_walk_price,
                               expected)

    def test_walking_price_combines_components(self):
        spot = 10
        risk_free = 0.0025
        time_step = 1
        price_steps = [0.25, 0.5]
        vol_steps = [0, 0]
        sut = self.TestStock(spot)
        expected = 18.8188125

        sut.walk_price(risk_free,
                       time_step,
                       price_steps,
                       vol_steps)
        self.assertAlmostEqual(sut.post_walk_price,
                               expected)

    def test_volatilities_determined_based_on_passed_in_vals(self):
        spot = 10
        risk_free = 0.0025
        time_step = 1
        price_steps = [0.25, 0.5]
        vol_steps = [20, 30]
        sut = self.TestStock(spot)

        sut.walk_price(risk_free,
                       time_step,
                       price_steps,
                       vol_steps)
        self.assertEqual(len(sut.vol_calls), 1)
        self.assertEqual(sut.vol_calls[0], vol_steps)


class ConstantVolatilityStockTestCase(unittest.TestCase):

    def test_constant_vol_is_a_stock(self):
        self.assertTrue(issubclass(ConstantVolatilityStock, Stock))

    def test_constant_vol_uses_constants(self):
        spot = 10
        vol = 2
        sut = ConstantVolatilityStock(spot, vol)

        res = sut.find_volatilities(0.01, [0 for _ in xrange(100)])

        self.assertEqual(list(res), [vol for _ in xrange(100)])


    def test_constant_vol_works_with_constant_vol(self):
        spot = 10
        risk_free = 0.0025
        time_step = 0.5
        price_steps = [0.25, 0.5]
        vol_steps = [20, 30]
        vol = 4
        sut = ConstantVolatilityStock(spot, vol)

        sut.walk_price(risk_free,
                       time_step,
                       price_steps,
                       vol_steps)
        self.assertAlmostEqual(sut.post_walk_price,
                               30.043765625)


class VariableVolatilityStockTestCase(unittest.TestCase):

    def test_var_vol_is_a_stock(self):
        self.assertTrue(issubclass(VariableVolatilityStock, Stock))

    def test_find_volatilities(self):
        spot = 10
        base_vol = 4
        k = 0.25
        g = 0.125
        vol_steps = [0.05, 0.0625]
        time_step = 0.5
        sut = VariableVolatilityStock(spot, base_vol, k, g)

        res = list(sut.find_volatilities(time_step, vol_steps))

        self.assertTrue(len(res), 2)
        v1, v2 = res
        self.assertAlmostEqual(v1, 4.0125)
        self.assertAlmostEqual(v2, 4.026586895018)

if __name__ == '__main__':
    unittest.main()
