from __future__ import division

import abc
import collections
import datetime
import functools
import itertools
import math
import numpy as np
import scipy.stats as ss

from mlmc import path, stock

class Option(object):

    __metaclass__ = abc.ABCMeta

    ''' A general representation of an option. '''

    def __init__(self, assets, risk_free, expiry, is_call):
        '''
        assets: list of underlying assets. Will probably be stocks in this framework.
        risk_free: the risk-free interest rate
        expiry: days until expiration of the option
        is_call: boolean. Whether the option is a call option or a put option
        '''
        self.assets = assets
        self.risk_free = risk_free
        self.expiry = expiry
        self.is_call = is_call

    @abc.abstractmethod
    def determine_payoff(self, *args, **kwargs):
        ''' Figure out the valuation of the option '''


class EuropeanStockOption(Option):

    ''' A stock option with a European payout '''

    def __init__(self, assets, risk_free, expiry, is_call, strike):
        '''
        assets: list of underlying assets. Will probably be stocks in this framework.
        risk_free: the risk-free interest rate
        expiry: days until expiration of the option
        is_call: boolean. Whether the option is a call option or a put option
        strike: the strike price of the option
        '''
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


class EuropeanSwaption(Option):

    ''' An exchange option with a European payout.'''

    def __init__(self, assets, risk_free, expiry, is_call):
        '''
        assets: list of underlying assets. Will probably be stocks in this framework.
        risk_free: the risk-free interest rate
        expiry: days until expiration of the option
        is_call: boolean. Whether the option is a call option or a put option
        '''

        if len(assets) != 2:
            raise ValueError('Requires two underlying assets')

        super(EuropeanSwaption, self).__init__(assets, risk_free, expiry, is_call)

    def determine_payoff(self, s1_final_spot, s2_final_spot, *args, **kwargs):
        v1, v2 = (s1_final_spot, s2_final_spot) if self.is_call else (s2_final_spot, s1_final_spot)
        return max(v1 - v2, 0)


class OptionSolver(object):

   ''' Given an option, will solve for the 'correct' price of the option '''

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def solve_option_price(self, option, return_stats=False):
        '''
        Actually solve the option price.
        option: an Option object. May need to be a specific type of option
        return_stats: boolean. Return not only the option price, but also associate statistics
        '''


class AnalyticEuropeanStockOptionSolver(OptionSolver):

    ''' A Black-Scholes stock option pricer. Only works for European stock options '''

    def solve_option_price(self, option):
        '''
        Actually solve the option price.
        option: an Option object. May need to be a specific type of option
        '''
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

        discount = math.exp(-risk_free * expiry)

        if option.is_call:
            S, d1, K, d2 = spot, d1, -strike, d2
        else:
            S, d1, K, d2 = -spot, -d1, strike, -d2

        return S * ss.norm.cdf(d1) + K * ss.norm.cdf(d2) * discount


class StatTracker(object):

    ''' Keeps track of running means and variances '''

    def __init__(self, discount):
        '''
        discount: the discount value that we will use to weigh all stats.
        '''
        self.discount = discount
        self.count = 0
        self.total = 0
        self.sum_of_squares = 0
        self.initial_val = None

    @property
    def variance(self):
        '''
        The running variance of the samples so far added
        '''
        if self.count in (0, 1):
            return float('inf')

        square_of_sum = self.total**2 / self.count
        variance = (self.sum_of_squares - square_of_sum) / (self.count - 1)
        return (self.discount * variance)

    @property
    def stdev(self):
        '''
        The running standard deviation of the samples so far added
        '''
        if self.count in (0, 1):
            return float('inf')

        return self.variance ** 0.5

    @property
    def mean(self):
        '''
        The running arithmetic mean of the samples so far added
        '''
        if self.count == 0:
            return float('nan')

        return self.discount * (self.total + self.initial_val*self.count) / self.count

    def add_sample(self, s):
        '''
        Add a sample to our set of samples for use in the
        running statistics
        '''
        if self.initial_val is None:
            self.initial_val = s

        self.count += 1
        diff = s - self.initial_val
        self.total += diff
        self.sum_of_squares += diff**2

    def get_interval_length(self, z_score):
        '''
        Determine the size of the confidence interval given a specific z-score
        z_score: float. The number of standard deviations away from the sample mean
                 we expect the population mean to fall into. 1.96 for 95% confidence.
        '''
        if self.count == 0:
            return float('inf')

        return self.stdev * self.count**(-0.5) * z_score


class NaiveMCOptionSolver(OptionSolver):

    '''
    Solve an option price using a simple Monte Carlo strategy
    of continued Euler-Maruyama paths until the resultant
    mean has an associated confidence interval shorter
    than the max_interval_length
    '''

    def __init__(self,
                 max_interval_length,
                 confidence_level=0.95,
                 rng_creator=None,
                 n_steps=None):
        '''
        max_interval_length: float. The longest the confidence interval may be
        confidence_level: float. The % chance the population mean falls within
                          the calculated confidence interval
        rng_creator: fn. No-arg function that returns a SampleCreator object
        n_steps: int. Number of steps per Euler-Maruyama path. Defaults to
                 option expiry normalized by the max_interval_length
        '''
        self.max_interval_length = max_interval_length
        self.confidence_level = confidence_level
        self.rng_creator = rng_creator
        self.n_steps = n_steps

    @property
    def confidence_level(self):
        ''' The confidence level of the solver '''
        return self._confidence_level

    @confidence_level.setter
    def confidence_level(self, value):
        self._confidence_level = value
        self._z_score = ss.norm.ppf(1 - 0.5*(1-self.confidence_level))

    @property
    def z_score(self):
        ''' The z score associated with the confidence level '''
        return self._z_score

    def _simulate_paths(self, option, n_steps, discount):
        stat_tracker = StatTracker(discount)
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
        '''
        Actually solve the option price.
        option: an Option object. May need to be a specific type of option
        return_stats: boolean. Return not only the option price, but also associate statistics
        '''
        expiry = option.expiry
        risk_free = option.risk_free
        discount = math.exp(-risk_free * expiry)

        n_steps = self.n_steps or int(math.floor(expiry / self.max_interval_length))

        tracker = self._simulate_paths(option, n_steps, discount)

        if return_stats:
            return tracker.mean, tracker.stdev, tracker.count, n_steps
        else:
            return tracker.mean


class LayeredMCOptionSolver(OptionSolver):

    ''' 
    Solve option price using a multi-level Monte Carlo (MLMC)
    strategy. There are multiple ways of doing this
    '''

    @abc.abstractmethod
    def run_levels(self, option, discount):
        '''
        Run the multiple levels of E-M paths

        option: Option. What we're looking to price
        discount: float. Discount factor of money in the future.
        '''

    def run_bottom_level(self, option, steps):
        '''
        Run the bottom level of E-M paths. Each of the E-M paths
        will have only one step

        option: Option. What we're looking to price
        steps: int. Totally irrelevant, used only because run_upper_levels
               requires it as part of the signature.
        '''
        result = path.create_simple_path(option.assets,
                                         option.risk_free,
                                         option.expiry,
                                         1,
                                         self.rng_creator)
        return option.determine_payoff(*result),

    def run_upper_level(self, option, steps):
        '''
        Run a non-bottom level of E-M paths. This comes out to
        running two paths, one with K times as many steps as the other

        option: Option. What we're looking to price
        steps: int. the number of steps for the path with fewer steps.
        '''
        result = path.create_layer_path(option.assets,
                                        option.risk_free,
                                        option.expiry,
                                        steps,
                                        self.rng_creator,
                                        K=self.level_scaling_factor)
        coarse, fine = zip(*result)
        payoff_coarse = option.determine_payoff(*coarse)
        payoff_fine = option.determine_payoff(*fine)

        return (payoff_fine - payoff_coarse),

    def run_level(self, option, L, n, *trackers):
        '''
        Run an individual level

        option: Option. What we're looking to price.
        L: int. The level we wish to run.
        n: int. The number of times we wish to run the level.
        *trackers: iterable of StatTrackers. Will be used to keep track of 
                   price, and possibly also time to run path.
        '''
        if L == 0:
            fn = self.run_bottom_level
            steps = 1
        else:
            fn = self.run_upper_level
            steps = self.level_scaling_factor ** (L - 1)
        for _ in xrange(n):
            for s, t in zip(fn(option, steps), trackers):
                t.add_sample(s)

    def solve_option_price(self, option, return_stats=False):
        '''
        Actually solve the option price.
        option: an Option object. May need to be a specific type of option
        return_stats: boolean. Return not only the option price, but also associate statistics
        '''
        expiry = option.expiry
        risk_free = option.risk_free
        discount = math.exp(-risk_free * expiry)
        trackers = self.run_levels(option, discount)

        if return_stats:
            means = [t.mean for t in trackers]
            variances = [t.variance for t in trackers]
            counts = [t.count for t in trackers]
            price = sum([t.mean for t in trackers])
            return (price, means, variances, counts)

        else:
            return sum([t.mean for t in trackers])


class SimpleLayeredMCOptionSolver(LayeredMCOptionSolver):

    '''
    The MLMC strategy should use a simple system for determining
    whether or not to continue, based on the empirical size of the
    highest level in comparison to the second-highest level. The number
    of paths run should simply be initially assumed as the same for all
    levels
    '''

    def __init__(self,
                 max_interval_length,
                 level_scaling_factor=4,
                 base_steps=1000,
                 rng_creator=None,
                 min_L=3):
        '''
        max_interval_length: float. Size of the error of the price
        level_scaling_factor: int. Ratio of steps in level l+1 to steps in level l
        base_steps: int. Number of paths to initially run per level
        rng_creator: no-arg function returning SampleCreator
        min_L: int. Starting number of levels to run
        '''
        self.max_interval_length = max_interval_length
        self.level_scaling_factor = max(level_scaling_factor, 2)
        self.base_steps = base_steps
        self.rng_creator = rng_creator
        self.min_L = min_L

    def _determine_additional_steps(self, option, trackers):
        find_dt = lambda l:  option.expiry / (self.level_scaling_factor ** l)
        tot = 2 * (self.max_interval_length ** -2) * sum(
            (t.variance/find_dt(L)) ** 0.5
            for L, t in enumerate(trackers)
        )
        ideal_ns = (
            tot * (t.variance * find_dt(L)) ** 0.5
            for L, t in enumerate(trackers)
        )
        return [
            int(math.ceil(max((ideal_n - t.count), 0)))
            for ideal_n, t in zip(ideal_ns, trackers)
        ]

    def _is_error_too_high(self, trackers):
        t1, t2 = trackers[-2:]

        empirical = max(abs(t1.mean) / self.level_scaling_factor, abs(t2.mean))
        estimated = ((self.level_scaling_factor - 1) * self.max_interval_length / (2**0.5))
        return estimated < empirical

    def run_levels(self, option, discount):
        trackers = [
            (self.base_steps, StatTracker(discount))
            for _ in xrange(self.min_L)
        ]

        while sum(n for n, _ in trackers) > 0:
            for L, (n, t) in enumerate(trackers):
                self.run_level(option, L, n, t)

            addl_steps = self._determine_additional_steps(
                option,
                [x[1] for x in trackers]
                )

            nt = []
            for L, (addl, (n, t)) in enumerate(zip(addl_steps, trackers)):
                self.run_level(option, L, addl, t)
                nt.append((0, t))

            trackers = nt

            if self._is_error_too_high([x[1] for x in trackers]):
                trackers.append((self.base_steps, StatTracker(discount)))

        return [t[1] for t in trackers]


class HeuristicLayeredMCOptionSolver(LayeredMCOptionSolver):

    '''
    The MLMC strategy should use a system for determining
    whether or not to continue based on the empirically-determined
    decay factors of the size of the layer means, layer variances and
    layer costs.
    '''

    def __init__(self,
                 target_mse,
                 rng_creator=None,
                 initial_n_levels=3,
                 level_scaling_factor=4,
                 initial_n_paths=5000,
                 alpha=None,
                 beta=None,
                 gamma=None):
        '''
        target_mse: float. Target mean standard error
        rng_cretor: no-arg function returning SampleCreator
        initial_n_levels: int. Number of levels to run initially. Must be >2
        level_scaling_factor: int. Ratio of steps in level l+1 to steps in level l
        initial_n_paths: int. Number of paths to run initially on the base level
        alpha: float. decay factor of the means of the level
        beta: float. decay factor of the variances of the level
        gamma: float. growth factor of the cost of the level
        '''
        self.target_mse = target_mse
        self.rng_creator = rng_creator
        self.initial_n_levels = max(initial_n_levels, 3)
        self.level_scaling_factor = max(level_scaling_factor, 2)
        self.initial_n_paths = initial_n_paths

        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma

    def cost_determined(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            d1 = datetime.datetime.now()
            res = fn(self, *args, **kwargs)
            d2 = datetime.datetime.now()

            delta = d2 - d1
            delta = delta.seconds + delta.microseconds*1e-6
            return delta, res[0]

        return wrapper

    run_bottom_level = cost_determined(LayeredMCOptionSolver.run_bottom_level)
    run_upper_level = cost_determined(LayeredMCOptionSolver.run_upper_level)

    def _determine_additional_n_values(self, trackers):
        overall = int(math.ceil(sum(
            (p.variance * c.mean)**0.5
            for _, p, c in trackers
        ) / (self.target_mse**2)))

        return [
            max(0, int(math.ceil(overall * (p.variance * c.mean)**0.5)) - p.count)
            for _, p, c in trackers
        ]

    def _find_coefficients(self, payoff_trackers, cost_trackers):
        A = np.array([[i, 1] for i, _ in enumerate(payoff_trackers, 1)])

        if self._alpha:
            alpha = self._alpha
        else:
            x = np.array([[np.log2(p.mean)] for p in payoff_trackers])
            alpha = max(0.5, -np.linalg.lstsq(A, x)[0][0])

        if self._beta:
            beta = self._beta
        else:
            x = np.array([[np.log2(p.variance)] for p in payoff_trackers])
            beta = max(0.5, -np.linalg.lstsq(A, x)[0][0])

        if self._gamma:
            gamma = self._gamma
        else:
            x = np.array([[np.log2(p.mean)] for p in cost_trackers])
            gamma = np.linalg.lstsq(A, x)[0][0]

        return alpha, beta, gamma

    def run_levels(self, option, discount):
        n_levels = self.initial_n_levels
        trackers = [
            (self.initial_n_paths, StatTracker(discount), StatTracker(1))
            for _ in xrange(n_levels)
        ]

        while sum(n for n, _, _ in trackers):
            for i, (n, payoff_tracker, cost_tracker) in enumerate(trackers):
                self.run_level(option, i, n, cost_tracker, payoff_tracker)

            addl_n_values = self._determine_additional_n_values(trackers)
            alpha, beta, gamma = self._find_coefficients(*zip(*((p, c) for (_, p, c) in trackers[1:])))

            trackers = [
                (addl_n, p, c)
                for addl_n, (_, p, c) in
                itertools.izip(addl_n_values, trackers)
            ]

            if all(n <= 0.01*p.count for n, p, _ in trackers):
                remaining_error = max(
                    (t.mean * 2**(alpha*i)) / (2**alpha - 1)
                    for i, (_, t, _) in enumerate(trackers[-2:], start=-2)
                )

                if remaining_error > (0.5**0.5) * self.target_mse:
                    guess_v = trackers[-1][1].variance  / (2^beta)
                    guess_c = trackers[-1][2].mean * (2 ** gamma)
                    term = (guess_v/guess_c) ** 0.5

                    base = sum((t.variance/c.cost)**0.5 for _, t, c in trackers)
                    base += term
                    guess_n = 2 * term * base / (self.target_mse**2)
                    trackers.append((guess_n, StatTracker(discount), StatTracker(1)))

        return [p for _, p, _ in trackers]
