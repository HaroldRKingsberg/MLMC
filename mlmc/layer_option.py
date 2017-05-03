from __future__ import division

from option import OptionSolver, NaiveMCOptionSolver

from mlmc import path, stock

class MLMCOptionSolver(OptionSolver):

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

    def _simulate_paths(self, option, discount, M=4):
        '''
        Args:
            M (int): constant to refine a coarse level to a fine level
        '''
        # notation using Giles (2008) paper
        # level (int): level l, which points to n_steps of 1 path at this level
        # L (int): highest level
        # Pl (float): an estimator of payoff using ONE path only, at level l 
        # Yl (float): estimator of E(P(l) - P(l-1)) using Nl paths
        # Nl (long): num paths to estimate Yl
        # fixedN (int): an arbitrary N to first estimate V(l), which then feeds V(l) into calculation of N(l)
        # Vl (float): variance of [P(l) - P(l-1)], estimated by fixedN paths
        # h (float): size of a time step

        
        
        # at level=0 then n_steps=1
        level, L, fixedN = 0, 0, 10000
        stat_tracker = NaiveMCOptionSolver.StatTracker(discount)
        # estimate at level 0, when l=0 then n_steps=1
        # step 2
        for i in xrange(fixedN):
            one_path = path.create_simple_path(option.assets,
                                               option.risk_free,
                                               option.expiry,
                                               M**level,
                                               self.rng_creator)
            payoff = option.determine_payoff(*one_path)
            stat_tracker.add_sample(payoff)
        V0 = stat_tracker.stdev**2        
        V_list = [V0]
        h0 = option.expiry
        h_list = [h0]
        # step 3
        N0 = 2 * self.max_interval_length**(-2) * (V0*h0)**0.5 * sum([(v/h)**0.5 for v, h in zip(V_list, h_list)])
        # step 4
        if N0 > fixedN:
            for i in xrange(N0 - fixedN):
                one_path = path.create_simple_path(option.assets,
                                               option.risk_free,
                                               option.expiry,
                                               1,
                                               self.rng_creator)
                payoff = option.determine_payoff(*one_path)
                stat_tracker.add_sample(payoff)
        Y0 = stat_tracker.mean

        level, L = 1, 1
        stat_tracker = NaiveMCOptionSolver.StatTracker(discount)
        for i in xrange(fixedN):
            one_path = path.create_layer_path(option.assets,
                                              option.risk_free,
                                              option.expiry,
                                              M**level,
                                              self.rng_creator,
                                              chunk_size=100000, 
                                              K=M)
            coarse_price = []
            fine_price = []
            for s1, s2 in one_path:
                coarse_price.append(s1)
                fine_price.append(s2)
            diff = option.determine_payoff(*fine_price) - option.determine_payoff(*coarse_price)
            stat_tracker.add_sample(diff)
        V1 = stat_tracker.stdev**2
        V_list.append(V1)
        h_list.append(option.expiry * M**(-level))