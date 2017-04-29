import copy
import numpy

import mlmc.random_numbers as random

def create_simple_path(stocks,
                       risk_free,
                       T,
                       n_steps,
                       rng_creator=None,
                       chunk_size=100000):
    '''
    Make each of the inputted stocks walk one path to final time t
    Each path consists of n_steps.
    To address memory issues, the simulation is run in chunks.
    Args:
        stocks (iterable): list of stocks that will walk
        risk_free (float): risk free rate driving stock drift
        T (float): the final time at end of a walk
        n_steps (long): number of steps along one path
        rng_creator (object): a maker of random number generator
        chunk_size (long): a walk of n_steps is done in chunks to address
            memory issues; chunk_size is the size of one chunk
    Returns:
    '''

    stocks = [copy.deepcopy(s) for s in stocks]

    # rng is a random number generator
    # if the user provides a maker of rng, then use that
    # if the maker rng_creator is not given, then default
    rng = rng_creator() if rng_creator else random.IIDSampleCreator(2*len(stocks))

    dt = float(T) / n_steps

    # walking n_steps is done in many chunks
    chunks = [chunk_size for _ in xrange(n_steps/chunk_size)]
    chunks.append(n_steps % chunk_size)

    # c is like one chunk; c is the size of one block (chunk) to be processed
    # simulate c steps at one round, out of n_steps
    for c in chunks:
        if not c:
            continue

        # samples (2D array) are like samples of dW or dZ for c mini steps
        # rng.size is like how many stocks we are simulating
        # num of rows of samples = rng.size, num columns = c
        samples = rng.create_sample(n_samples=c, time_step=dt)
        # len(samples) := num rows;
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
