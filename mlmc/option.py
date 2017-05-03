from __future__ import division

import abc
import collections
import itertools
import math
import numpy as np
import scipy.stats as ss

from mlmc import path, stock

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
    def solve_option_price(self, option, return_stats=False):
        return None


class AnalyticEuropeanStockOptionSolver(OptionSolver):

    def solve_option_price(self, option):
        underlying = option.assets[0]
        spot = underlying.spot
        vol = underlying.vol
        risk_free = option.risk_free
        expiry = option.expiry
        strike = option.strike

        log_diff = math.log(spot / strike)
        vt = 0.5 * vol**2
        denom = vol * math.sqrt(expiry)

        d1 = (log_diff + (risk_free + vt)*expiry) / denom
        d2 = (log_diff + (risk_free - vt)*expiry) / denom
        # F = spot * math.exp(expiry * risk_free)

        discount = math.exp(-risk_free * expiry)
        
        if option.is_call:
            S, d1, K, d2 = spot, d1, -strike, d2
        else:
            S, d1, K, d2 = -spot, -d1, strike, -d2
        
        return S * ss.norm.cdf(d1) + K * ss.norm.cdf(d2) * discount


class NaiveMCOptionSolver(OptionSolver):

    class StatTracker(object):

        def __init__(self, discount):
            self.discount = discount
            self.count = 0
            self.total = 0
            self.sum_of_squares = 0
            self.initial_val = None

        @property
        def stdev(self):
            if self.count in (0, 1):
                return float('inf')

            square_of_sum = self.total**2 / self.count
            variance = (self.sum_of_squares - square_of_sum) / (self.count - 1)
            return (self.discount * variance) ** 0.5

        @property
        def mean(self):
            if self.count == 0:
                return float('nan')

            return self.discount * (self.total + self.initial_val*self.count) / self.count

        def add_sample(self, s):
            if self.initial_val is None:
                self.initial_val = s

            self.count += 1
            diff = s - self.initial_val
            self.total += diff
            self.sum_of_squares += diff**2

        def get_interval_length(self, z_score):
            if self.count == 0:
                return float('inf')

            return self.stdev * self.count**(-0.5) * z_score


    def __init__(self, max_interval_length, confidence_level=0.95, rng_creator=None):
        self.max_interval_length = max_interval_length
        self.confidence_level = confidence_level
        self.rng_creator = rng_creator

    @property
    def confidence_level(self):
        return self._confidence_level

    @confidence_level.setter
    def confidence_level(self, value):
        self._confidence_level = value
        self._z_score = ss.norm.ppf(1 - 0.5*(1-self.confidence_level))

    @property
    def z_score(self):
        return self._z_score

    def _simulate_paths(self, option, n_steps, discount):
        stat_tracker = self.StatTracker(discount)
        cnt = itertools.count()

        while next(cnt) < 10 or stat_tracker.get_interval_length(self.z_score) > self.max_interval_length:
            result = path.create_simple_path(option.assets,
                                             option.risk_free,
                                             option.expiry,
                                             n_steps,
                                             self.rng_creator)
            payoff = option.determine_payoff(*result)
            stat_tracker.add_sample(payoff)

        return stat_tracker

    def solve_option_price(self, option, return_stats=False):
        expiry = option.expiry
        risk_free = option.risk_free
        discount = math.exp(-risk_free * expiry)

        n_steps = int(math.floor(expiry / self.max_interval_length))

        tracker = self._simulate_paths(option, n_steps, discount)

        if return_stats:
            return tracker.mean, tracker.stdev, tracker.count, n_steps
        else:
            return tracker.mean
