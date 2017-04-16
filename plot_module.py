from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def plot_hist(samples, true_mean, chart_title, x_label, y_label):
    '''
    Plot a histogram (normalized into density) of the samples 
    Args:
        samples: array of size n which are n samples
        true_mean: a scalar for the true mean of the random variable being sampled
        chart_title: string for title of the graph 
        x_label: string for x axis label
        y_label: string for y axis label
    Return:
        plot a normalized histogram with overlay normal distribution density curve 
    '''
    plt.figure()
    # the bar histogram
    _, bin_edges, _ = plt.hist(samples, bins='auto', normed=True, facecolor='green')
    # overlay a normal density 
    samples_mean = np.asscalar(np.mean(samples))
    samples_stdev = np.asscalar(np.std(samples))
    normal_density = mlab.normpdf(bin_edges, samples_mean, samples_stdev)
    plt.plot(bin_edges, normal_density, 'r--')
    # show the true mean as a vertical line
    plt.vlines(true_mean, 0, np.amax(normal_density), linestyles='dashed')
    # put on labels 
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(chart_title)
    plt.show()