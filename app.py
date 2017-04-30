from mlmc import option, stock

s = stock.ConstantVolatilityStock(100, .1)
e = option.EuropeanCall(s, .05, 120, 1)
e.euler_asset_walk(10000, 20)
