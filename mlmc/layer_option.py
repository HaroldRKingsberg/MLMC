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
                M=4 is used in Giles (2008) paper
        '''
        # notation using Giles (2008) paper
        # level (int): level l, which points to n_steps of 1 path at this level
        # L (int): highest level
        # Pl (float): an estimator of payoff using ONE path only, at level l 
        # Yl (float): estimator of E(P(l) - P(l-1)) using Nl paths
        # Nl (long): num paths to estimate Yl
        # fixedN (int): an arbitrary N to first estimate V(l), 
        # which then feeds V(l) into calculation of N(l)
        # Vl (float): variance of [P(l) - P(l-1)], first from fixedN paths
        # h (float): size of a time step at one level
        
        epsilon = self.max_interval_length

        # at level=0 then n_steps=1
        level, L, fixedN = 0, 0, 10000
        stat_tracker = NaiveMCOptionSolver.StatTracker(discount)        
        # step 2
        for i in xrange(fixedN):
            one_path = path.create_simple_path(option.assets,
                                               option.risk_free,
                                               option.expiry,
                                               M**L,
                                               self.rng_creator)
            payoff = option.determine_payoff(*one_path)
            stat_tracker.add_sample(payoff)        
        V_list = [stat_tracker.stdev**2]
        h_list = [option.expiry / (M**L)]
        # step 3
        N0 = 2 * epsilon**(-2) * (V0*h0)**0.5 * sum([
            (v/h)**0.5 for v, h in zip(V_list, h_list)])
        N_list = [N0]
        # step 4
        if N0 > fixedN:
            for i in xrange(N0 - fixedN):
                one_path = path.create_simple_path(option.assets,
                                                   option.risk_free,
                                                   option.expiry,
                                                   M**L,
                                                   self.rng_creator)
                payoff = option.determine_payoff(*one_path)
                stat_tracker.add_sample(payoff)        
        Y_list = [stat_tracker.mean]
        # has to update V0 for later use
        V_list[0] = stat_tracker.stdev**2

        level, L = 1, 1
        stat_tracker = NaiveMCOptionSolver.StatTracker(discount)
        for i in xrange(fixedN):
            one_path = path.create_layer_path(option.assets,
                                              option.risk_free,
                                              option.expiry,
                                              M**L,
                                              self.rng_creator,
                                              chunk_size=100000, 
                                              K=M)
            coarse_price = []
            fine_price = []
            for s1, s2 in one_path:
                coarse_price.append(s1)
                fine_price.append(s2)
            diff = option.determine_payoff(*fine_price) - \
                   option.determine_payoff(*coarse_price)
            stat_tracker.add_sample(diff)
        V_list.append(stat_tracker.stdev**2)
        h_list.append(option.expiry / M**L)
        N = 2 * epsilon**(-2) * (V_list[-1] * h_list[-1])**0.5 * sum([(v/h)**0.5 for v, h in zip(V_list, h_list)])
        N_list.append(N)
        if N > fixedN:
            for i in xrange(N - fixedN):
                one_path = path.create_layer_path(option.assets,
                                              option.risk_free,
                                              option.expiry,
                                              M**L,
                                              self.rng_creator,
                                              chunk_size=100000, 
                                              K=M)
                coarse_price = []
                fine_price = []
                for s1, s2 in one_path:
                    coarse_price.append(s1)
                    fine_price.append(s2)
                diff = option.determine_payoff(*fine_price) - \
                       option.determine_payoff(*coarse_price)
                stat_tracker.add_sample(diff)
        Y_list.append(stat_tracker.mean)
        V_list[-1] = stat_tracker.stdev**2

        while L < 2 or (Y_list[-1] - Y_list[-2] / M) < ((M**2 - 1) * epsilon / (2**0.5)):
            L += 1
            stat_tracker = NaiveMCOptionSolver.StatTracker(discount)
            for i in xrange(fixedN):
                one_path = path.create_layer_path(option.assets,
                                                  option.risk_free,
                                                  option.expiry,
                                                  M**L,
                                                  self.rng_creator,
                                                  chunk_size=100000, 
                                                  K=M)
                coarse_price = []
                fine_price = []
                for s1, s2 in one_path:
                    coarse_price.append(s1)
                    fine_price.append(s2)
                diff = option.determine_payoff(*fine_price) - \
                       option.determine_payoff(*coarse_price)
                stat_tracker.add_sample(diff)
            V_list.append(stat_tracker.stdev**2)
            h_list.append(option.expiry / M**L)
            N = 2 * epsilon**(-2) * (V_list[-1] * h_list[-1])**0.5 * sum([(v/h)**0.5 for v, h in zip(V_list, h_list)])
            N_list.append(N)
            if N > fixedN:
                for i in xrange(N - fixedN):
                    one_path = path.create_layer_path(option.assets,
                                                option.risk_free,
                                                option.expiry,
                                                M**L,
                                                self.rng_creator,
                                                chunk_size=100000, 
                                                K=M)
                    coarse_price = []
                    fine_price = []
                    for s1, s2 in one_path:
                        coarse_price.append(s1)
                        fine_price.append(s2)
                    diff = option.determine_payoff(*fine_price) - 
                        option.determine_payoff(*coarse_price)
                    stat_tracker.add_sample(diff)
            Y_list.append(stat_tracker.mean)
            V_list[-1] = stat_tracker.stdev**2
        # I don't know how to return a stat tracker here???
        # the output is Yhat and variance(Yhat) below
        Y = sum(Y_list)
        var_Y = sum([v/n for v, n in zip(V_list, N_list)])