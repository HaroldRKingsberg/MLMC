from mlmc import option, stock, random_numbers
from tests.analytic import black_scholes
import math
spot = 100
vol = .1
risk_free = .05
strike = 120
expiry = 1



b = black_scholes(S0=spot, K=strike, r=risk_free, q=0, sigma=vol, T=expiry, option_type="call")

for i in xrange(5):
    s = stock.ConstantVolatilityStock(spot, vol)
    e = option.EuropeanCall([s], risk_free, strike, expiry)
    p = e.euler_asset_walk(20000, 5000)
    exp_p = math.exp(-risk_free*expiry)*p.mean()
    resid = abs(b-exp_p)
    print(resid)
