import copy
import numpy

import mlmc.src.random_numbers as random

def create_simple_path(stocks,
                       risk_free,
                       t,
                       n_steps,
                       rng_creator=None,
                       chunk_size=100000):

    stocks = [copy.deepcopy(s) for s in stocks]
    rng = rng_creator() if rng_creator else random.IIDSampleCreator(2*len(stocks))
    dt = float(t) / n_steps

    chunks = [chunk_size for _ in xrange(n_steps/chunk_size)]
    chunks.append(n_steps % chunk_size)

    for c in chunks:
        if not c:
            continue

        samples = rng.create_sample(n_samples=c, time_step=dt)
        interval = len(samples) / len(stocks)

        for i, s in enumerate(stocks):
            s.walk_price(risk_free,
                         dt,
                         *samples[i*interval:(i+1)*interval])

    return [s.post_walk_price for s in stocks]


def create_layer_path(stocks,
                      risk_free,
                      t,
                      n_steps,
                      rng_creator=None,
                      chunk_size=100000,
                      K=2):
    stocks = [
        (copy.deepcopy(s), copy.deepcopy(s))
        for s in stocks
    ]

    rng = rng_creator() if rng_creator else random.IIDSampleCreator(2*len(stocks))

    dt = float(t) / n_steps
    dt_sub = dt / K
    chunks = [chunk_size for _ in xrange(n_steps/chunk_size)]
    chunks.append(n_steps % chunk_size)

    for c in chunks:
        if not c:
            continue

        samples = rng.create_sample(n_samples=c*K,
                                    time_step=dt_sub)
        interval = len(samples) / len(stocks)

        for i, (s1, s2) in enumerate(stocks):
            subs = samples[i*interval:(i+1)*interval]
            fulls = numpy.array([
                numpy.array([s[i:i+K].sum() for i in xrange(0, c*K, K)])
                for s in subs
            ])

            s1.walk_price(risk_free, dt, *fulls)
            s2.walk_price(risk_free, dt_sub, *subs)


    return [
        (s1.post_walk_price, s2.post_walk_price)
        for s1, s2 in stocks
    ]


def calculate(task):
    return task[0](*task[1:])


def main():
    import multiprocessing
    from mlmc.src.stock import ConstantVolatilityStock
    pool = multiprocessing.Pool(4)
    stock = ConstantVolatilityStock(10, 0.1)

    x = pool.map(
        calculate,
        [
            [create_simple_path] + [[stock, stock], 0.01, 1, 100000]
            for _ in xrange(100)
        ]
    )

    import pprint
    pprint.pprint(x)

    x = pool.map(
        calculate,
        [
            [create_layer_path] + [[stock, stock], 0.01, 1, 50000]
            for _ in xrange(100)
        ]
    )
    pprint.pprint(x)


if __name__ == '__main__':
    main()
