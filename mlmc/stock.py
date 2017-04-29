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
        Simulate the stock to walk one full path over multiple steps
        Args:
            risk_free (float): the rate in the deterministic term of dS
            time_step (float): size of a mini time step (dt)
            price_steps (list): a 1D iterable of the Brownian increments driving the diffusion term in dS
            vol_steps (list): a 1D iterable of the Brownian increments driving stochastic variance in the Heston model
        Returns:
            float: the price S(T) after walking one full simulated path
        '''

        # vol_steps supposedly contains the Brownian dZ for stochastic Heston
        # for Heston, use dZ in vol_steps to find the path of stochastic vol
        # but if the Brownian dZ for vol is not given by user then vol is const
        # then we only need vol_steps to contain marks of the time steps        
        vol_steps = price_steps if vol_steps is None else vol_steps
        
        # vols is a Generator of the sigma for each of the time step interval
        vols = self.find_volatilities(time_step, vol_steps)

        # start the walk from current price of the stock
        price = self.post_walk_price

        # each step uses both diffusion in dS and the vol over that one step
        # pstep is the Brownian increment inside dS
        # vol is the sigma in the diffusion term of dS over one time step
        for pstep, vol in itertools.izip(price_steps, vols):
            # det_term = risk_free * time_step
            # sto_term = (vol**0.5) * pstep
            # price += (det_term + sto_term) * price
            price = price * math.exp((risk_free - 0.5 * vol**2) * time_step + vol * pstep)

        self.post_walk_price = price
        return price


class ConstantVolatilityStock(Stock):
    '''
    A stock with constant volatility 
    Attributes:
        spot (float): the current spot price 
        post_walk_price (float): the price at T after walking one full path
        _vol (float): the constant vol in the diffusion term
    '''

    def __init__(self, spot, vol):
        super(ConstantVolatilityStock, self).__init__(spot)
        self._vol = vol

    def find_volatilities(self, time_step, vol_steps):
        '''
        Return the volatility values over multiple time steps
        This is a generator function that returns a Generator of vol
        Args:
            time_step (float): size of a mini time step 
            vol_steps (list): here the vol is constant so vol_steps contain the marks of time steps that vol will undertake
        Returns:
            generator: a generator to generate vol on the fly, over time steps
        '''
        # given vol is constant, the generator just yields this const vol 
        # generator will be exhausted after running the num time steps required
        return itertools.repeat(self._vol, len(vol_steps))


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
            generator: a generator to generate vol on the fly, over time steps
        '''
        # V(0) is initial variance which is vol(0)**2
        variance = self._base_vol**2

        # use the increment dZ from vol_steps to simulate V(t)
        for dZ in vol_steps:
            # the deterministic term in dV(t)
            det_term = self._kappa * (self._theta - max(0, variance)) * time_step
            # the diffusion term in dV(t)
            sto_term = self._gamma * (max(0, variance)**0.5) * dZ
            # get the next value of V(t)
            variance += (det_term + sto_term)
            # the full truncation method is used, if V(t) < 0 then
            # only use the positive part of V(t) as the vol
            # thus yield the vol which is V(t)**0.5
            yield max(0, variance)**0.5