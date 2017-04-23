import copy
import numpy

import mlmc.src.random_numbers as random

def create_euro_option_path(stock,
                            strike,
                            risk_free,
                            t,
                            n_steps,
                            is_call,
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

    if is_call:
        prop = stock.post_walk_price - strike
    else:
        prop = strike - stock.post_walk_price

    return max(0, prop)


def create_euro_option_layer_path(stock,
                                  strike,
                                  risk_free,
                                  t,
                                  n_steps,
                                  is_call,
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

    if is_call:
        prop = (s2.post_walk_price - strike) - (s1.post_walk_price - strike)
    else:
        prop = (strike - s2.post_walk_price) - (strike - s1.post_walk_price)

    return max(0, prop)


def calculate(task):
    return task[0](*task[1:])


if __name__ == '__main__':
    import multiprocessing
    from mlmc.src.stock import ConstantVolatilityStock
    pool = multiprocessing.Pool(4)
    stock = ConstantVolatilityStock(10, 0.0)
    '''
    x = pool.map(
        calculate,
        [
            [create_euro_option_path] + [stock, 5, 0.01, 1, 100000, True]
            for _ in xrange(100)
        ]
    )
    
    import numpy
    print numpy.array(x).mean()
    '''
    x = pool.map(
        calculate,
        [
            [create_euro_option_layer_path] + [stock, 5, 0.01, 1, 50000, True]
            for _ in xrange(100)
        ]
    )
    print x
    
