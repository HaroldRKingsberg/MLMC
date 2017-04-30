from scipy.stats import norm
import math
import numpy as np
from analytic import black_scholes

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
    
    npaths = target_epsilon**(-2)
    step_size = target_epsilon    
    nsteps = math.floor(T / step_size)
    
    simulated_payoffs = self.euler_asset_walk(nsteps, npaths)
    return np.mean(simulated_payoffs) * math.exp(-r*T)

def check_vanilla_mc(self, target_stdev, confidence_level, num_samples=1000):
    samples_option_values = [vanilla_montecarlo(target_stdev, confidence_level) for i in xrange(num_samples)]

    S0 = self.assets.spot
    K = self.strike
    r = self.risk_free
    q = 0
    sigma = self.assets.vol
    T = self.expirary_time
    option_type = 'call'
    true_value = black_scholes(S0, K, r, q, sigma, T, option_type)

    z_score = norm.ppf(1 - (1 - confidence_level) / 2)
    target_epsilon = z_score * target_stdev
    checks = [abs(x - true_value) > target_epsilon for x in samples_option_values]
    return (sum(checks) / len(checks)) <= (1 - confidence_level)