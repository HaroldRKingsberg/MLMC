from mlmc.src import path, stock
import multiprocessing
import pprint

pool = multiprocessing.Pool(4)
s = stock.ConstantVolatilityStock(10, 0)

x = pool.map(
    path.calculate,
    [
        [path.create_simple_path] + [[s], 0.01, 1, 100000]
        for _ in xrange(100)
    ]
)

pprint.pprint(x)
