import itertools
import math
import unittest

from mlmc.option import (EuropeanStockOption,
                         AnalyticEuropeanStockOptionSolver,
                         NaiveMCOptionSolver)
from mlmc.stock import ConstantVolatilityStock
from analytic import black_scholes

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
        # is_call = True
        option = EuropeanStockOption([stock], risk_free, expiry, True, strike)

        solver = AnalyticEuropeanStockOptionSolver()

        call_value = black_scholes(spot, strike, risk_free, 0, vol, expiry, 'call')
        self.assertAlmostEqual(call_value, 6.04008812972)

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

        put_value = black_scholes(spot, strike, risk_free, 0, vol, expiry, 'put')
        self.assertAlmostEqual(put_value, 10.6753248248)

        self.assertAlmostEqual(solver.solve_option_price(option),
                               10.6753248248)


class NaiveMCOptionSolverTestCase(unittest.TestCase):

    def test_z_score(self):
        sut = NaiveMCOptionSolver(0, confidence_level=0.95)
        self.assertAlmostEqual(sut.z_score, 1.96, 2)

    # The following test can be commented out because it takes
    # over a minute to run.
    def test_put_option(self):
        spot = 100
        strike = 110
        risk_free = 0.05
        expiry = 1
        vol = 0.2

        stock = ConstantVolatilityStock(spot, vol)
        option = EuropeanStockOption([stock], risk_free, expiry, False, strike)

        interval = 0.1
        expected = 10.6753248248
        lower_bound = expected - interval
        upper_bound = expected + interval
        solver = NaiveMCOptionSolver(interval)
        n_runs = 20
        in_bound_count = 0

        for i in xrange(n_runs):
            price = solver.solve_option_price(option)

            if lower_bound <= price <= upper_bound:
                in_bound_count += 1

        self.assertGreaterEqual(in_bound_count, 0.95*n_runs)


if __name__ == '__main__':
    unittest.main()
