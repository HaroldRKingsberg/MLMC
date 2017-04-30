from mlmc import path, stock
import multiprocessing
import pprint

class Option(object):

    def __init__(self, assets, risk_free, strike, expirary_time):

        self.assets = assets
        self.risk_free = risk_free
        self.strike = strike
        self.expirary_time = expirary_time


class EuropeanCall(Option):

    def __init__(self, asset, risk_free, strike, expirary_time):
        super(EuropeanCall, self).__init__(asset, risk_free, strike, expirary_time)
        self.payoff_func = lambda x: max(x - strike, 0)

    def euler_asset_walk(self, nsteps, npaths):



        pool = multiprocessing.Pool(1)

        x = pool.map(
            path.calculate,
            [
                [path.create_simple_path] + [[self.assets], self.risk_free, self.expirary_time, nsteps]
                for _ in xrange(npaths)
            ]
        )

        payoffs = [self.payoff_func(price) for price in x]

        print(payoffs)
