import copy
import numpy

import mlmc.src.random_numbers as random

def create_simple_path(stock,
                       risk_free,
                       t,
                       n_steps,
                       rng=None,
                       chunk_size=100000):

    stock = copy.deepcopy(stock)
    rng = copy.deepcopy(rng) if rng else random.IIDSampleCreator(2)
    dt = float(t) / n_steps

    chunks = [chunk_size for _ in xrange(n_steps/chunk_size)]
    chunks.append(n_steps % chunk_size)

    for c in chunks:
        if not c:
            continue

        stock.walk_price(risk_free,
                         dt,
                         *rng.create_sample(n_samples=c,
                                            time_step=dt))

    return stock.post_walk_price


def create_layer_path(stock,
                      risk_free,
                      t,
                      n_steps,
                      rng=None,
                      chunk_size=100000,
                      K=2):
    s1 = copy.deepcopy(stock)
    s2 = copy.deepcopy(stock)

    rng = copy.deepcopy(rng) if rng else random.IIDSampleCreator(2)

    dt = float(t) / n_steps
    dt_sub = dt / K
    chunks  = [chunk_size for _ in xrange(n_steps/chunk_size)]
    chunks.append(n_steps % chunk_size)

    for c in chunks:
        if not c:
            continue

        subs = rng.create_sample(n_samples=c*K,
                                 time_step=dt_sub)
        fulls = numpy.array([
            numpy.array([s[i:i+K].sum() for i in xrange(0, c*K, K)])
            for s in subs
        ])

        s2.walk_price(risk_free, dt_sub, *subs)
        s1.walk_price(risk_free, dt, *fulls)

    return (s2.post_walk_price, s1.post_walk_price)


def calculate(task):
    return task[0](*task[1:])


if __name__ == '__main__':
    import multiprocessing
    from mlmc.src.stock import ConstantVolatilityStock
    pool = multiprocessing.Pool(4)
    stock = ConstantVolatilityStock(10, 0.1)
                            
    x = pool.map(
        calculate,
        [
            [create_simple_path] + [stock, 0.01, 1, 100000]
            for _ in xrange(100)
        ]
    )
    
    import numpy
    print numpy.array(x).mean()

    x = pool.map(
        calculate,
        [
            [create_layer_path] + [stock, 0.01, 1, 50000]
            for _ in xrange(100)
        ]
    )
    print x
    
