import itertools
import math
import unittest

from mlmc.option import (EuropeanStockOption,
                         AnalyticEuropeanStockOptionSolver)
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


if __name__ == '__main__':
    unittest.main()
