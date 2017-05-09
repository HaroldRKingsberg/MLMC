from mlmc import option, stock, random_numbers
from tests.analytic import black_scholes
import math
import numpy as np
import matplotlib.pyplot as plt
import os


def create_graph(asset):

    def plot_vars_and_means(asset, ax1, ax2):
        '''
        Produces plots for upper left and upper right.
        '''

        solver = option.SimpleLayeredMCOptionSolver(
            max_interval_length=.01,
            base_steps=10000,
            min_L=5
            )

        (means, variances, counts) = solver.solve_option_price(asset, True)
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

        for i, e in enumerate(epsilon_list):
            print("NEW TRIAL")
            solver = option.SimpleLayeredMCOptionSolver(
                max_interval_length=e,
                base_steps=1000,
                min_L=3
                )

            (means, variances, counts) = solver.solve_option_price(asset, True)

            for l, (m, v, c) in enumerate(zip(means, variances, counts)):
                print(l, m, v, c)

            layers = [i for i in range(len(counts))]
            ax3.plot(layers, counts, ls="--",marker='x',label="e = %s" %e)
        ax3.set_yscale('log')
        ax3.set_xlabel("N_l")
        ax3.set_ylabel("l")
        ax3.legend(loc='upper right')


    fig, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2,2)
    plot_vars_and_means(asset, ax1, ax2)
    plot_Npaths_for_epsilon(asset, ax3)
    plt.tight_layout()
    plt.savefig(os.getcwd()+"/figs/graph_1.png")
    plt.show()

def main():
    spot = 1
    strike = 1
    risk_free = 0.05
    expiry = 1
    vol = 0.2



    s = stock.ConstantVolatilityStock(spot, vol)
    o = option.EuropeanStockOption(s, risk_free, expiry, True, strike)
    create_graph(o)

if __name__ == '__main__':
    main()
