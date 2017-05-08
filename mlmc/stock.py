import abc
import itertools
import math

class Stock(object):
    '''
    Abstract base class for a stock object
    Attributes:
        spot (float): the current spot price
        post_walk_price (float): the price at T after walking one full path
    '''

    __metaclass__ = abc.ABCMeta

    def __init__(self, spot):
        self.spot = spot
        self.post_walk_price = spot

    @abc.abstractmethod
    def find_volatilities(self, time_step, vol_steps):
        '''
        Return the volatility values over multiple time steps
        This is a generator function that returns a generator of vol
        Args:
            time_step (float): size of a mini time step (dt)
            vol_steps (list): a 1D iterable of the Brownian increments driving stochastic variance in the Heston model
        Returns:
            generator: a generator to generate vol on the fly, over time steps
        '''

    def walk_price(self, risk_free, time_step, price_steps, vol_steps=None):
        '''
        Simulate the stock to walk one full path over multiple steps using geometric Brownian motion
        Args:
            risk_free (float): the rate in the deterministic term of dS
            time_step (float): size of a mini time step (dt)
            price_steps (list): a 1D iterable of the Brownian increments driving the diffusion term in dS
            vol_steps (list): a 1D iterable of the Brownian increments driving stochastic variance in the Heston model. If not provided, default behavior is to use the price_steps.

        Returns:
            float: the price S(T) after walking one full simulated path.
        '''
        vol_steps = price_steps if vol_steps is None else vol_steps
        vols = self.find_volatilities(time_step, vol_steps)

        lprice = math.log(self.post_walk_price)

        # for dW, sigma in itertools.izip(price_steps, vols):
        #     t1 = (risk_free - (0.5 * sigma**2)) * time_step
        #     t2 = sigma * dW
        #     lprice += (t1 + t2)
        # self.post_walk_price = math.exp(lprice)

        price = self.post_walk_price
        for dW, sigma in itertools.izip(price_steps, vols):
            price += time_step*risk_free*price + sigma*price*dW

        self.post_walk_price = price
        return self.post_walk_price


class ConstantVolatilityStock(Stock):
    '''
    A stock with constant volatility
    Attributes:
        spot (float): the current spot price
        post_walk_price (float): the price at T after walking one full path
        vol (float): the constant vol in the diffusion term
    '''

    def __init__(self, spot, vol):
        super(ConstantVolatilityStock, self).__init__(spot)
        self.vol = vol

    def find_volatilities(self, time_step, vol_steps):
        '''
        Return the volatility values over multiple time steps
        This is a generator function that returns a Generator of vol
        Args:
            time_step (float): size of a mini time step
            vol_steps (list): here the vol is constant so vol_steps contain the marks of time steps that vol will undertake
        Returns:
            generator: returns the stock's volatility n times, where n is the length of vol_steps
        '''
        return itertools.repeat(self.vol, len(vol_steps))


class VariableVolatilityStock(Stock):
    '''
    A stock with stochastic volatility based on Heston model
    Attributes:
        spot (float): the current spot price
        post_walk_price (float): the price at T after walking one full path
        _base_vol (float): the starting point of the vol at t = 0. The square of _base_vol is the starting variance V(0)
        _kappa (float): the mean reversion speed in Heston SDE for dV(t)
        _theta (float): the mean reversion level in Heston SDE for dV(t)
        _gamma (float): the constant diffusion term in Heston SDE for dV(t)
    '''

    def __init__(self, spot, base_vol, kappa, theta, gamma):
        super(VariableVolatilityStock, self).__init__(spot)

        self.vol = base_vol
        self._base_vol = base_vol
        self._kappa = kappa
        self._theta = theta
        self._gamma = gamma

    def find_volatilities(self, time_step, vol_steps):
        '''
        Return the volatility values over multiple time steps
        This is a generator function that returns a Generator of vol
        Args:
            time_step (float): size of a mini time step
            vol_steps (list): a 1D iterable of the Brownian increments driving stochastic variance in the Heston model
        Returns:
            generator: a random walk of the volatility using the full truncation method. Thus, if we are left in a situation where the next step would lead us to a negative variance, the step instead goes to zero.
        '''
        volatility = max(0, self.vol)
        variance = self._base_vol**2

        for dZ in vol_steps:
            drift = self._kappa * (self._theta - variance) * time_step
            diffusion = self._gamma * volatility * dZ
            variance = variance + drift + diffusion
            volatility = max(0, variance) ** 0.5
            self.vol = volatility
            yield volatility
