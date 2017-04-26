from __future__ import division
import numpy as np
import matplotlib.pylab as plt


class SDE(object):

    def __init__(self, drift, diffusion):
        '''
        drift: f(X) R^m -> R^m
        diffusion: g(X) R^m -> R^(m,d)
        '''

        self.drift = drift
        self.diffusion = diffusion

    def one_step(self, X, dt):
        f_x = self.drift(X)
        g_x = self.diffusion(X)
        dw = dt**.5*np.random.randn(g_x.shape[1], 1)
        return dt*f_x + np.matmul(g_x, dw)


    def EM_Solver(self, N, X0, t_start=0, t_finish=1):

        dt = (t_finish - t_start) / N
        X_old = X0
        paths = np.zeros((X0.shape[0], N+1))

        for step in range(N):
            paths[:, step] = X_old
            X_new = X_old + self.one_step(X_old, dt)
            X_old = X_new

        paths[:, N] = X_new
        return paths

    def plot_paths(self, paths):

        N = paths.shape[1]
        x_ticks = [i for i in range(N)]
        for i in range(paths.shape[0]):
            plt.plot(x_ticks, paths[i,:])

        plt.show()


class testProblem(SDE):

    def __init__(self, c1, c2):
        self.drift = lambda X: c1*X
        self.diffusion = lambda X: c2*X


def main():

    test = testProblem(10,2)
    path = test.EM_Solver(100, np.array([[.2]]))
    test.plot_paths(path)









if __name__ == '__main__':
    main()
