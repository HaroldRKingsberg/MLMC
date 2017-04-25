import abc
import itertools

class Stock(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, spot):
        self.spot = spot
        self.post_walk_price = spot

    @abc.abstractmethod
    def find_volatilities(self, time_step, vol_steps):
        ''' Return the volatility values over time '''

    def walk_price(self, risk_free, time_step, price_steps, vol_steps=None):
        vol_steps = price_steps if vol_steps is None else vol_steps
        vols = self.find_volatilities(time_step, vol_steps)
        price = self.post_walk_price

        for pstep, vol in itertools.izip(price_steps, vols):
            det_term = risk_free * time_step
            sto_term = (vol**0.5) * pstep
            price += (det_term + sto_term) * price

        self.post_walk_price = price
        return price


class ConstantVolatilityStock(Stock):

    def __init__(self, spot, vol):
        super(ConstantVolatilityStock, self).__init__(spot)
        self._vol = vol

    def find_volatilities(self, time_step, vol_steps):
        return itertools.repeat(self._vol, len(vol_steps))


class VariableVolatilityStock(Stock):

    def __init__(self, spot, base_vol, k, g):
        super(VariableVolatilityStock, self).__init__(spot)

        self._base_vol = base_vol
        self._k = k
        self._g = g

    def find_volatilities(self, time_step, vol_steps):
        vol = self._base_vol

        for s in vol_steps:
            det_term = self._k * (self._base_vol - vol) * time_step
            sto_term = self._g * (vol**0.5) * s
            vol += (det_term + sto_term)
            yield vol