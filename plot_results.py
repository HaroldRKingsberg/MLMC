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

def calc_total_numsteps(counts, level_scaling_factor=4):
    '''
    Helper function to calculate total num steps taken
    Args:
        counts(list): list of N(l) num paths at level l
    '''
    C = counts[0]
    for l in range(1, len(counts)):
        C += counts[l]*(level_scaling_factor**l + level_scaling_factor**(l-1))
    return C
    
def plot_numpaths(one_option, ax3, ax4, epsilon_list, str_solver_type):
    '''
    Two lower graphs for simple layer solver or heuristic solver
    Args:
        str_solver_type (string): either "simple" or "heuristic"
    '''
    # this is num steps * epsilon**2
    ml_cost_list = []
    naive_cost_list = []
    for i, epsilon in enumerate(epsilon_list):
        print("NEW TRIAL")        
    
        if str_solver_type == "simple":
            layer_solver = option.SimpleLayeredMCOptionSolver(epsilon)
        elif str_solver_type == "heuristic":
            layer_solver = option.HeuristicLayeredMCOptionSolver(epsilon)
        else:
            raise ValueError("type is 'simple' or 'heuristic'")        
        (option_price, means, variances, counts) = layer_solver.solve_option_price(one_option, True)
        epsilon_scaled_cost = calc_total_numsteps(counts, layer_solver.level_scaling_factor) * epsilon**2
        ml_cost_list.append(epsilon_scaled_cost)
        ax3.plot(range(len(counts)), counts, ls="--",marker='x',label="e = %s" % epsilon)
        
        naive_solver = option.NaiveMCOptionSolver(epsilon)
        option_value, pricer_stdev, n_paths, n_steps = naive_solver.solve_option_price(one_option, True)
        naive_cost_list.append(n_paths * n_steps * epsilon**2)
        
    ax3.set_yscale('log')
    ax3.set_xlabel("level")
    ax3.set_ylabel("N(l)")
    ax3.legend(loc='best')
    
    ax4.plot(epsilon_list, ml_cost_list, marker='x', label='MLMC')
    ax4.plot(epsilon_list, naive_cost_list, ls="--", marker='x', label='Std MC')
    ax4.set_xlabel("epsilon")
    ax4.set_ylabel("cost x square(epsilon)")
    ax4.legend(loc='best')
        
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
    
    heston_swaption = option.EuropeanSwaption([hvs_1, hvs_2], risk_free, expiry, is_call)    
    
    ROOT_DIR = os.getcwd()
    epsilon = 0.5
    
    # Case 1: simple layer solver, heston vol, swaption
    filename = ROOT_DIR + '/figs/simple_layer_heston_swap'    
    fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2,2)
    
    simple_layer_solver = option.SimpleLayeredMCOptionSolver(epsilon)
    plot_log_var_mean(heston_swaption, simple_layer_solver, ax1, ax2)
    
    epsilon_list = [0.01, 0.05, 0.1, 0.25, 0.5]
    plot_numpaths(heston_swaption, ax3, ax4, epsilon_list, 'simple')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    