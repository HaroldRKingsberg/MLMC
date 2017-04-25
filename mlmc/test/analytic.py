from __future__ import division
import unittest 
import numpy as np
import scipy.stats as ss
from analytical_solutions import black_scholes


def black_scholes(S0, K, r, q, sigma, T, option_type):
    '''
    Gives the Black Scholes option prices for vanilla European calls and puts 
    Args:
        S0: stock spot price at t = 0
        K: option strike price 
        r: risk free rate 
        q: continuous dividend rate 
        sigma: constant stock volatility 
        T: option time to expiry
        option_type: string 'call' or 'put'
    Return: 
        value: price of the vanilla European option 
    '''
    d1 = (np.log(S0 / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r - q - sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    F = S0 * np.exp((r - q) * T)
    if option_type == 'call':
        value = np.exp(-r * T) * (F * ss.norm.cdf(d1) - K * ss.norm.cdf(d2))
    elif option_type == 'put':
        value = np.exp(-r * T) * (-F * ss.norm.cdf(-d1) + K * ss.norm.cdf(-d2))
    else:
        raise ValueError('option_type input is bad; input either "call" or "put"')
    return value


def margrabe(S1, S2, q1, q2, sig1, sig2, T, rho):
    '''
    Gives the closed form Margrabe's formula for a European exchange option
    The option payoff at expiry time T is max(0, S1(T)-S2(T))
    Args:
        S1, S2: the current price of 2 stocks at time 0
        q1, q2: the respective dividend rate of each stock 
        sig1, sig2: the respective constant volatility of each stock
        T: expiry time of the option 
        rho: correlation coefficient of the 2 Brownian motions driving the 2 stocks
    Return:
        value: price of the European exchange option
    '''
    if abs(rho) > 1: 
        raise ValueError('rho input must be between [-1, 1]')
    sig = np.sqrt(sig1**2 + sig2**2 - 2 * sig1 * sig2 * rho)
    d1 = (np.log(S1 / S2) + (q2 - q1 + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    value = np.exp(-q1 * T) * S1 * ss.norm.cdf(d1) - np.exp(-q2 * T) * S2 * ss.norm.cdf(d2)
    return value


class AnalyticalSolutionsTestCase(unittest.TestCase):

    def test_black_scholes(self):
        S0, K = 100, 110
        r, q, T, sigma = 0.05, 0.02, 1, 0.2
        self.assertAlmostEqual(black_scholes(S0, K, r, q, sigma, T, 'call'), 5.18858, places=4)
        self.assertAlmostEqual(black_scholes(S0, K, r, q, sigma, T, 'put'), 11.80395, places=4)


if __name__ == '__main__':
    unittest.main()
