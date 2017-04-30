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
    Get the post walk price of each of the inputted stocks
    Each stock walks one path to final time T
    The post walk price is returned without changing the stock itself
    This path simulation is vanilla Monte Carlo for Euler Maruyama
    To address memory issues, the simulation is run in chunks.
    Args:
        stocks (iterable): list of stocks that will walk
        risk_free (float): risk free rate driving stock drift
        T (float): the final time at end of a walk
        n_steps (long): number of steps along one path
        rng_creator (callable): a no-arg function that will return an
            object implementing the SampleCreator interface. Default
            is a function that returns an IIDSampleCreator that outputs
            both dW and dZ.
        chunk_size (long): a walk of n_steps is done in chunks to address 
            memory issues; chunk_size is the size of one chunk

    Returns:
        list: the post walk price of each stock
    '''

    stocks = [copy.deepcopy(s) for s in stocks]
    rng = rng_creator() if rng_creator else random.IIDSampleCreator(2*len(stocks))

    dt = float(T) / n_steps

    chunks = [chunk_size for _ in xrange(n_steps/chunk_size)]
    chunks.append(n_steps % chunk_size)

    for c in chunks:
        if not c:
            continue
        
        samples = rng.create_sample(n_samples=c, time_step=dt)
        interval = rng.size / len(stocks)

        for i, s in enumerate(stocks):
            # samples[i*interval] are the dW
            # samples[i*interval + 1] are the dZ
            s.walk_price(risk_free,
                         dt,
                         *samples[i*interval:(i+1)*interval])

    return [s.post_walk_price for s in stocks]


def create_layer_path(stocks,
                      risk_free,
                      T,
                      n_steps,
                      rng_creator=None,
                      chunk_size=100000,
                      K=2):
    '''    
    Each stock walks one path in the finer level, and one path on the coarser level, to final time T
    The post walk price is returned without changing the stock itself
    This path simulation is MultiLevel Monte Carlo for Euler Maruyama
    To address memory issues, the simulation is run in chunks.
    Args:
        stocks (iterable): list of stocks that will walk
        risk_free (float): risk free rate driving stock drift
        T (float): the final time at end of a walk
        n_steps (long): number of steps along one path
        rng_creator (callable): a no-arg function that will return an
            object implementing the SampleCreator interface. Default
            is a function that returns an IIDSampleCreator that outputs
            both dW and dZ.
        chunk_size (long): a walk of n_steps is done in chunks to address 
            memory issues; chunk_size is the size of one chunk
        K (int): for level L in MLMC, the interval [0,T] is partitioned into K**L intermediate time steps

    Returns:
        list: the post walk price of each stock
    '''

    stocks = [
        (copy.deepcopy(s), copy.deepcopy(s))
        for s in stocks
    ]

    rng = rng_creator() if rng_creator else random.IIDSampleCreator(2*len(stocks))

    # dt is for the coarser level, dt_sub for finer level
    dt = float(T) / n_steps
    dt_sub = dt / K

    chunks = [chunk_size for _ in xrange(n_steps/chunk_size)]
    chunks.append(n_steps % chunk_size)

    for c in chunks:
        if not c:
            continue
        
        # first, samples := dW and dZ are the finer level
        samples = rng.create_sample(n_samples=c*K,
                                    time_step=dt_sub)
        interval = rng.size / len(stocks)

        # s1 walks the coarse level, s2 walks the fine level
        for i, (s1, s2) in enumerate(stocks):
            subs = samples[i*interval:(i+1)*interval] # finer level
            fulls = numpy.array([
                numpy.array([s[i:i+K].sum() for i in xrange(0, c*K, K)])
                for s in subs
            ]) # coarser level

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
