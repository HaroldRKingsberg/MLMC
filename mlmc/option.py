from __future__ import division

import abc
import collections
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
    def solve_option_price(self, option):
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
        F = spot * math.exp(expiry * risk_free)

        discount = math.exp(-risk_free * expiry)

        if option.is_call:
            F, d1, strike, d2 = F, d1, -strike, d2
        else:
            F, d1, strike, d2 = -F, -d1, strike, -d2

        return discount * (F * ss.norm.cdf(d1) + strike * ss.norm.cdf(d2))


class NaiveMCOptionSolver(OptionSolver):

    def __init__(self, desired_stdev, confidence_level=0.95, rng_creator=None):
        self.desired_stdev = desired_stdev
        self.confidence_level = confidence_level
        self.rng_creator = rng_creator

    @property
    def confidence_interval_spread(self):
        z = ss.norm.ppf(1 - 0.5*(1-self.confidence_level))
        return z * self.desired_stdev

    @property
    def n_paths(self):
        return int(self.confidence_interval_spread ** -2)

    def find_simulated_mean(self, option, n_steps):
        arg_list = [
            option.assets,
            option.risk_free,
            option.expiry,
            n_steps,
            self.rng_creator
        ]

        total_payoff = sum(
            option.determine_payoff(*path.create_simple_path(*arg_list))
            for _ in xrange(self.n_paths)
        )

        return total_payoff / self.n_paths

    def solve_option_price(self, option):
        expiry = option.expiry
        risk_free = option.risk_free
        discount = math.exp(-risk_free * expiry)

        n_steps = int(math.floor(expiry / self.confidence_interval_spread))

        simulated_mean = self.find_simulated_mean(option, n_steps)
        return simulated_mean * math.exp(-risk_free * expiry)
