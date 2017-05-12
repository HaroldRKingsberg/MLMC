from mlmc import option, stock, random_numbers
from tests.analytic import black_scholes
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from decimal import Decimal
import matplotlib

def create_graph(asset, filename):

    def plot_vars_and_means(asset, ax1, ax2):
        '''
        Produces plots for upper left and upper right.
        '''

        solver = option.SimpleLayeredMCOptionSolver(
            max_interval_length=.01,
            base_steps=10000,
            min_L=5
            )

        (price, means, variances, counts) = solver.solve_option_price(asset, True)
        for l, (m, v, c) in enumerate(zip(means, variances, counts)):
            print(l, m, v, c)

        # means and variances actually contain V(p(l) - p(l-1))
        layer_means = np.array(means).cumsum()
        layer_vars = np.array(variances).cumsum()

        # take log of base 4, for plotting
        variances = [math.log(v,4) for v in variances]
        layer_vars = [math.log(v,4) for v in layer_vars]
        means = [math.log(abs(m),4) for m in means]
        layer_means = [math.log(abs(m),4) for m in layer_means]

        ax1.plot([1,2,3,4,5], layer_vars, marker ='x', label="P_l")
        ax1.plot([2,3,4,5], variances[1:], ls="--",marker='x', label="P_l - P_(l-1)")
        ax1.set_xlabel("l")
        ax1.set_ylabel("Log_M Variance")
        ax1.legend()

        ax2.plot([1,2,3,4,5], layer_means, marker ='x', label="P_l")
        ax2.plot([2,3,4,5], means[1:], ls="--",marker='x', label="P_l - P_(l-1)")
        ax2.set_xlabel("l")
        ax2.set_ylabel("Log_M |mean|")
        ax2.legend()

    def plot_Npaths_for_epsilon(asset, ax3):

        epsilon_list = [.001, .0005, .0002, .0001]
        #epsilon_list = [.01, .005, .002, .001]

        def calc_comp_cost(counts, level_scaling_factor=4):
            C = counts[0]
            for l in range(1, len(counts)):
                C += counts[l]*(level_scaling_factor**l + level_scaling_factor**(l-1))
            return C

        cost_list = []
        for i, e in enumerate(epsilon_list):
            print("NEW TRIAL")
            solver = option.SimpleLayeredMCOptionSolver(
                max_interval_length=e,
                base_steps=1000,
                min_L=3
                )

            (price, means, variances, counts) = solver.solve_option_price(asset, True)

            for l, (m, v, c) in enumerate(zip(means, variances, counts)):
                print(l, m, v, c)
            cost_list.append(calc_comp_cost(counts))
            layers = [i for i in range(len(counts))]
            ax3.plot(layers, counts, ls="--",marker='x',label="e = %s" %e)

        ax3.set_yscale('log')
        ax3.set_xlabel("l")
        ax3.set_ylabel("N_l")
        ax3.legend(loc='upper right')
        ax4.plot(epsilon_list, cost_list, marker='x', label='MLMC')
        ax4.set_xlabel("epsilon")
        ax4.set_ylabel("Cost (total MC steps)")
        ax4.legend(loc='upper right')


    fig, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2,2)
    plot_vars_and_means(asset, ax1, ax2)
    plot_Npaths_for_epsilon(asset, ax3)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    #plt.show()

def create_hist(opt, n_trials=500, n_bins=20, interval=.005):

    exact_solver = option.AnalyticEuropeanStockOptionSolver()
    exact_sol = exact_solver.solve_option_price(opt)

    lower_bound = exact_sol - interval
    upper_bound = exact_sol + interval

    solver = option.SimpleLayeredMCOptionSolver(
        max_interval_length=interval,
        base_steps=1000
        )

    approx_sols = np.zeros((n_trials,1))
    in_bound_count = 0
    for i in range(n_trials):
        approx_sol = solver.solve_option_price(opt, False)
        if lower_bound <= approx_sol <= upper_bound:
            in_bound_count +=1
        approx_sols[i] = approx_sol


    mean, sigma = np.mean(approx_sols), np.std(approx_sols)
    conf_int = stats.norm.interval(.95, loc=mean, scale=sigma)
    fig, ax = plt.subplots()
    ax.hist(approx_sols, bins=50)
    ax.axvline(x=conf_int[0],color='r',ls='--',label='95% CI')
    ax.axvline(x=conf_int[1],color='r',ls='--')
    ax.axvline(x=lower_bound,color='g',ls='--',label='epsilon +/-%f' %interval)
    ax.axvline(x=upper_bound,color='g',ls='--')
    ax.set_xlabel("Option Price")
    ax.set_ylabel("Count")
    plt.legend(loc='upper right', bbox_to_anchor=(1.02, 1), borderaxespad=0)

    plt.title("%i trials; Sample Empirical Var.%.2E\nNo. Samples in epsilon interval %i" %(n_trials, Decimal(np.var(approx_sols)), in_bound_count))

    plt.show()


def main():
    #params for constant vol stock 1
    spot_1 = 1
    vol_1 = 0.2

    #params for constant vol stock 2
    spot_2 = .8
    vol_2 = .3

    #params for heston vol stock 1
    spot_3 = 1
    vol_3 = .2
    kappa_3 = .1
    theta_3 = .05
    gamma_3 = .03

    #params for heston vol stock 2
    spot_4 = .8
    vol_4 = .3
    kappa_4 = .08
    theta_4 = .06
    gamma_4 = .04

    #params for options
    strike = 1
    risk_free = 0.05
    expiry = 1

    cvs_1 = stock.ConstantVolatilityStock(spot_1, vol_1)
    cvs_2 = stock.ConstantVolatilityStock(spot_2, vol_2)

    hvs_1 = stock.VariableVolatilityStock(spot_3, vol_3, kappa_3, theta_3, theta_3)
    hvs_2 = stock.VariableVolatilityStock(spot_4, vol_4, kappa_4, theta_4, theta_4)

    o1 = option.EuropeanStockOption(cvs_1, risk_free, expiry, True, strike)

    o2 = option.EuropeanSwaption([cvs_1, cvs_2], risk_free, expiry, True)

    o3 = option.EuropeanSwaption([hvs_1, hvs_2], risk_free, expiry, True)

    ROOT_DIR = os.getcwd()
    filename = None
    #filename = ROOT_DIR+"/figs/constant_vol_call.png"
    # create_graph(o1, filename)

    #filename = ROOT_DIR+"/figs/constant_vol_exchange.png"
    # create_graph(o2, filename)

    #filename = ROOT_DIR+"/figs/heston_vol_exchange.png"
    # create_graph(o3, filename)

    create_hist(o1, n_trials=1000)



if __name__ == '__main__':
    main()
