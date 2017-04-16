from __future__ import division
import unittest 
import numpy as np
from analytical_solutions import black_scholes

class TestAnalyticalSolutions(unittest.TestCase):
    def test_black_scholes(self):
        S0, K = 100, 110
        r, q, T, sigma = 0.05, 0.02, 1, 0.2
        self.assertAlmostEqual(black_scholes(S0, K, r, q, sigma, T, 'call'), 5.18858, places=4)
        self.assertAlmostEqual(black_scholes(S0, K, r, q, sigma, T, 'put'), 11.80395, places=4)

if __name__ == '__main__':
    unittest.main()