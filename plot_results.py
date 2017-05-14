import os
import math
import numpy as np
import matplotlib.pyplot as plt
from mlmc import option, stock

def plot_log_var_mean(one_option, layer_solver, ax1, ax2):
    '''
    This is the upper left and upper right plot    
    '''
    # M is the factor to refine one level to the next
    M = int(layer_solver.level_scaling_factor)
    (option_price, means, variances, counts) = layer_solver.solve_option_price(one_option, True)
    L = len(variances)
    
    # mean and var of P(L) where L is highest level and P(L) = sum of Y(l)    
    max_level_vars = np.array(variances).cumsum()
    max_level_means = np.array(means).cumsum()
    
    # log of var and mean of Y(l)
    log_vars = [math.log(v, M) for v in variances]
    log_means = [math.log(abs(m), M) for m in means]
    
    # log of var and mean of P(L)
    log_max_level_vars = [math.log(v, M) for v in max_level_vars]
    log_max_level_means = [math.log(abs(m), M) for m in max_level_means]
    
    ax1.plot(range(1,L+1), log_max_level_vars, marker ='x', label="P(l)")
    ax1.plot(range(2,L+1), log_vars[1:], ls="--",marker='x', label="P(l) - P(l-1)")
    ax1.set_xlabel("level")
    ax1.set_ylabel("Log_M Variance")
    ax1.legend(loc='best')

    ax2.plot(range(1,L+1), log_max_level_means, marker ='x', label="P(l)")
    ax2.plot(range(2,L+1), log_means[1:], ls="--", marker='x', label="P(l) - P(l-1)")
    ax2.set_xlabel("level")
    ax2.set_ylabel("Log_M |mean|")
    ax2.legend(loc='best')

def main():
    spot_1 = 100
    vol_1 = 0.2
    kappa_1 = 0.15
    theta_1 = 0.25    
    gamma_1 = 0.1
    
    spot_2 = 50
    vol_2 = 0.4
    kappa_2 = 0.35
    theta_2 = 0.5
    gamma_2 = 0.2
    
    risk_free = 0.05
    expiry = 1.5
    is_call = True
    
    hvs_1 = stock.VariableVolatilityStock(spot_1, vol_1, kappa_1, theta_1, theta_1)
    hvs_2 = stock.VariableVolatilityStock(spot_2, vol_2, kappa_2, theta_2, theta_2)
    
    swaption = option.EuropeanSwaption([hvs_1, hvs_2], risk_free, expiry, is_call)    
    
    ROOT_DIR = os.getcwd()
    epsilon = 0.5
    
    filename = ROOT_DIR + '/figs/simple_layer_heston_swap'
    simple_layer_solver = option.SimpleLayeredMCOptionSolver(epsilon)
    
    fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2,2)
    plot_log_var_mean(swaption, simple_layer_solver, ax1, ax2)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    