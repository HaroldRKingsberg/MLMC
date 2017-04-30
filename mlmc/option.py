from __future__ import division
from mlmc import path, stock
import multiprocessing
import pprint
import numpy as np
from scipy.stats import norm
from tests.analytic import black_scholes
import math

class Option(object):

    def __init__(self, assets, risk_free, strike, expirary_time):

        self.assets = assets
        self.risk_free = risk_free
        self.strike = strike
        self.expirary_time = expirary_time


class EuropeanCall(Option):

    def __init__(self, asset, risk_free, strike, expirary_time):
        super(EuropeanCall, self).__init__(asset, risk_free, strike, expirary_time)
        self.payoff_func = lambda x: max(x - strike, 0)

    def euler_asset_walk(self, nsteps, npaths):

        # weirdness with random numbers generated in multiprocessing???
        # pool = multiprocessing.Pool(1)
        #
        #
        #
        # x = pool.map(
        #     path.calculate,
        #     [
        #         [path.create_simple_path] + [[self.assets], self.risk_free, self.expirary_time, nsteps]
        #         for _ in xrange(npaths)
        #     ]
        # )

        x = np.zeros((npaths, 1))
        for i in xrange(npaths):
            x[i,:] = path.create_simple_path(
                [self.assets],
                self.risk_free,
                self.expirary_time,
                nsteps
                )

        payoffs = np.array([self.payoff_func(price[0]) for price in x])
        return payoffs

    def vanilla_montecarlo(self, target_stdev, confidence_level):
        '''
        Args:
            target_stdev(float): target stdev of the estimator of the option price
                then epsilon will be like 1.98 * target_stdev
            confidence_level(float): between [0.5, 1] e.g. 0.9 for 90% confidence
        '''
        T = self.expirary_time
        r = self.risk_free

        z_score = norm.ppf(1 - (1 - confidence_level) / 2)
        target_epsilon = z_score * target_stdev

        npaths = int(target_epsilon**(-2))
        step_size = target_epsilon
        nsteps = int(math.floor(T / step_size))

        simulated_payoffs = self.euler_asset_walk(nsteps, npaths)
        return np.mean(simulated_payoffs) * math.exp(-r*T)

    def check_vanilla_mc(self, target_stdev, confidence_level, num_samples=1000):
        samples_option_values = [self.vanilla_montecarlo(target_stdev, confidence_level) for i in xrange(num_samples)]

        S0 = self.assets.spot
        K = self.strike
        r = self.risk_free
        q = 0
        sigma = self.assets._vol
        T = self.expirary_time
        option_type = 'call'
        true_value = black_scholes(S0, K, r, q, sigma, T, option_type)

        z_score = norm.ppf(1 - (1 - confidence_level) / 2)
        target_epsilon = z_score * target_stdev
        checks = [abs(x - true_value) > target_epsilon for x in samples_option_values]
        print(checks)
        return (sum(checks) / len(checks)) <= (1 - confidence_level), sum(checks), len(checks)
