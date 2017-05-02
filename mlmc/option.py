from __future__ import division

import abc
import collections
import multiprocessing
import numpy as np


from mlmc import path, stock
from scipy.stats import norm
import math

class Option(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, assets, risk_free, expiry, is_call):
        self.assets = assets
        self.risk_free = risk_free
        self.expiry = expiry
        self.is_call = is_call

    @abc.abstractmethod
    def determine_payoff(self, *args, **kwargs):
        ''' Figure out the valuation of the option '''


class EuropeanStockOption(Option):

    def __init__(self, assets, risk_free, expiry, is_call, strike):
        if isinstance(assets, collections.Iterable):
            assets = assets[:1]
            if not isinstance(assets[0], stock.Stock):
                raise TypeError("Requires an underlying stock")
        elif isinstance(assets, stock.Stock):
            assets = [assets]
        else:
            raise TypeError("Requires an underlying stock")

        super(EuropeanStockOption, self).__init__(assets, risk_free, expiry, is_call)
        self.strike = strike

    def determine_payoff(self, final_spot, *args, **kwargs):
        v1, v2 = (final_spot, self.strike) if self.is_call else (self.strike, final_spot)
        return max(v1 - v2, 0)


class OptionSolver(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def solve_option_price(self, option):
        return None


class AnalyticEuropeanStockOptionSolver(OptionSolver):

    def solve_option_price(self, option):
        underlying = option.assets[0]
        spot = underlying.spot
        vol = underlying.vol
        risk_free = option.risk_free
        expiry = option.expiry

        t1 = np.log(spot / option.strike)
        t2 = vol * np.sqrt(expiry)

        d1 = t1 + (risk_free + vol**2/2) * expiry
        d2 = t1 + (risk_free - vol**2/2) * expiry
        F = spot * np.exp(expiry * (risk_free))

        if option.is_call:
            F, d1, d2, K = F, d1, d2, -strike
        else:
            F, d1, d2, K = -F, -d1, -d2, strike

        val = F * ss.norm.cdf(d1) * K * ss.norm.cdf(d2)
        return np.exp(-risk_free * expiry) * val
        


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

