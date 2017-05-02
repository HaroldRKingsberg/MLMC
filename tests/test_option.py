import itertools
import math
import unittest

from mlmc.option import (EuropeanStockOption,
                         AnalyticEuropeanStockOptionSolver,
                         NaiveMCOptionSolver)
from mlmc.stock import ConstantVolatilityStock

class EuropeanStockOptionTestCase(unittest.TestCase):

    def test_valuation_for_call(self):
        stock = ConstantVolatilityStock(10, 1)
        sut = EuropeanStockOption([stock], 0.05, 1, True, 10)
        self.assertEqual(sut.determine_payoff(6), 0)
        self.assertEqual(sut.determine_payoff(15), 5)

    def test_valuation_for_put(self):
        stock = ConstantVolatilityStock(10, 1)
        sut = EuropeanStockOption([stock], 0.05, 1, False, 10)
        self.assertEqual(sut.determine_payoff(6), 4)
        self.assertEqual(sut.determine_payoff(15), 0)


class AnalyticEuropeanStockOptionSolverTestCase(unittest.TestCase):

    def test_option_pricing(self):
        spot = 100
        strike = 110
        risk_free = 0.05
        expiry = 1
        vol = 0.2

        stock = ConstantVolatilityStock(spot, vol)
        option = EuropeanStockOption([stock], risk_free, expiry, True, strike)

        solver = AnalyticEuropeanStockOptionSolver()
        self.assertAlmostEqual(solver.solve_option_price(option),
                               6.04008812972)

    def test_put_option(self):
        spot = 100
        strike = 110
        risk_free = 0.05
        expiry = 1
        vol = 0.2

        stock = ConstantVolatilityStock(spot, vol)
        option = EuropeanStockOption([stock], risk_free, expiry, False, strike)

        solver = AnalyticEuropeanStockOptionSolver()
        self.assertAlmostEqual(solver.solve_option_price(option),
                               10.6753248248)


class NaiveMCOptionSolverTestCase(unittest.TestCase):

    def test_confidence_spread(self):
        solver = NaiveMCOptionSolver(0.1, 0.9)
        self.assertAlmostEqual(solver.confidence_interval_spread,
                               .1645,
                               4)

    def test_put_option(self):
        spot = 100
        strike = 110
        risk_free = 0.05
        expiry = 1
        vol = 0.2

        stock = ConstantVolatilityStock(spot, vol)
        option = EuropeanStockOption([stock], risk_free, expiry, False, strike)

        expected = 10.6753248248
        solver = NaiveMCOptionSolver(0.1)
        spread = solver.confidence_interval_spread
        n_runs = 1000
        res = [solver.solve_option_price(option) for _ in xrange(n_runs)]
        import pprint
        pprint.pprint(sorted(res))
        import numpy as np
        print np.std(np.array(res))
    

if __name__ == '__main__':
    unittest.main()
