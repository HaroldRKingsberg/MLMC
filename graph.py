from mlmc import option, stock, random_numbers
from tests.analytic import black_scholes
import math
import numpy as np
import matplotlib.pyplot as plt
import os

spot = 1
strike = 1
risk_free = 0.05
expiry = 1
vol = 0.2
interval = 0.1

s = stock.ConstantVolatilityStock(spot, vol)
o = option.EuropeanStockOption(s, risk_free, expiry, True, strike)
solver = option.SimpleLayeredMCOptionSolver(interval,base_steps=10000,max_L=5)


def generate_log_variance(solver):
    trackers = solver.solve_option_price(o)
    for (n, t) in trackers:
        print(n, t.mean, t.variance, t.count)

    # E[P(l) - P(l-1)]
    pl_means_deltas = np.array([t.mean for (n,t) in trackers])

    # Var[P(l) - P(l-1)]
    pl_vars_deltas = np.array([t.variance for (n,t) in trackers])

    # E[P(l)] = P_0 + sum(E[P(l) -P(l-1)]) for l = 1,2,...
    pl_means = pl_means_deltas.cumsum()

    # Var[P(l)] = P_0 + sum(var[P(l) -P(l-1)]) for l = 1,2,...
    pl_vars = pl_vars_deltas.cumsum()

    # take log of base 4, for plotting
    pl_vars_deltas = [math.log(v,4) for v in pl_vars_deltas]
    pl_vars = [math.log(v,4) for v in pl_vars]
    pl_means_deltas = [math.log(abs(m),4) for m in pl_means_deltas]
    pl_means = [math.log(abs(m),4) for m in pl_means]

    fig, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2,2)
    ax1.plot([1,2,3,4,5], pl_vars, marker ='x', label="P_l")
    ax1.plot([2,3,4,5], pl_vars_deltas[1:], ls="--",marker='x', label="P_l - P_(l-1)")
    ax1.set_xlabel("l")
    ax1.set_ylabel("Log_M Variance")
    ax1.legend()

    ax2.plot([1,2,3,4,5], pl_means, marker ='x', label="P_l")
    ax2.plot([2,3,4,5], pl_means_deltas[1:], ls="--",marker='x', label="P_l - P_(l-1)")
    ax2.set_xlabel("l")
    ax2.set_ylabel("Log_M |mean|")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.getcwd()+"/figs/graph_1.png")
    plt.show()

def main():
    generate_log_variance(solver)

if __name__ == '__main__':
    main()
