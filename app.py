from mlmc import option, stock, random_numbers
from tests.analytic import black_scholes
import math

spot = 100
vol = .1
risk_free = .05
strike = 120
expiry = 1
eps=.05

s = stock.ConstantVolatilityStock(spot, vol)
e = option.EuropeanCall(s, risk_free, strike, expiry)
count = 0
print(e.check_vanilla_mc(target_stdev=eps, confidence_level=.51, num_samples=100))
