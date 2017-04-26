import abc

from numpy import array, identity, linalg, matrix, random
import numpy as np

class SampleCreator(object):
    '''
    Abstract base class for a simulator of one path each for an ensemble of random variables (eg several Brownian motions W1, W2, etc)
    Attributes:
        size (int): the number of random variables
    '''
    # indicate that this is an abstract base class (abc)
    __metaclass__ = abc.ABCMeta

    def __init__(self, size):
        self.size = size

    @abc.abstractmethod
    def create_sample(self, n_samples=1, time_step=1, *args):
        '''
        Generate a single path for each random variable (number of random variables, eg number of Brownian motions, is self.size). One path consists of n_samples steps
        Args:
            n_samples (int): the number of steps along a single path 
            time_step (float): the size of a mini time step
        Returns: 
            ndarray: a 2D array (num rows: self.size, num cols: n_samples). Each row is one single path for one random variable
        '''


class IIDSampleCreator(SampleCreator):
    '''
    Simulator of one path each for an ensemble of independent random variables
    Attributes:
        size (int): the number of independent random variables (eg the number of independent Brownian motions W1, W2, etc)
        distro (method of numpy.random): the distribution from which to take one step and form one random path
    '''
    def __init__(self, size, distro=random.normal):
        self.distro = distro
        super(IIDSampleCreator, self).__init__(size)

    def create_sample(self, n_samples=1, time_step=1, *args):
        '''
        Generate a single path for each random variable (number of random variables, eg number of Brownian motions, is self.size). One path consists of n_samples steps
        Args:
            n_samples (int): the number of steps along a single path 
            time_step (float): the size of a mini time step. For Brownian motion, time_step is variance of the normal dist
        Returns: 
            ndarray: a 2D array (num rows: self.size, num cols: n_samples). Each row is one single path for one random variable
        '''
        # use list comprehension to generate a list of normal random numbers
        # each sample has a mean = 0 and variance = time_step
        # the number of samples is n_samples, for steps along a single path 
        # the list is then converted into a numpy array
        return array([
            self.distro(scale=time_step**0.5, size=n_samples)
            for _ in xrange(self.size)
        ])


class CorrelatedSampleCreator(IIDSampleCreator):
    '''
    Simulator of one path each for an ensemble of correlated random variables (eg correlated Brownian motions). When the variables together take one time step, their steps are correlated by a given correlation matrix
    Attributes:
        size (int): the number of correlated random variables (eg the number of correlated Brownian motions W1, W2, etc)
        distro (method of numpy.random): the distribution from which to take one step and form one random path
        _corr_matrix (ndarray): correlation matrix of the steps being taken along correlated paths by the random variables
        _ts_transforms (dict): a dict mapping one time_step to a transformation matrix C where C * C.T = the covariance matrix corresponding to the size of that time_step
    '''

    def __init__(self, corr_matrix, scales=None, distro=random.normal):
        self._set_corr_matrix(corr_matrix)
        size = len(corr_matrix) # len(ndarray) returns num of rows
        self._ts_transforms = {}

        super(CorrelatedSampleCreator, self).__init__(size=size,
                                                      distro=distro)

    def _set_corr_matrix(self, cm):
        '''
        Set the corr matrix of self to cm after checking conditions
        Args:
            cm (ndarray): a symmetrix positive definite correlation matrix
        Returns:
            Set self._corr_matrix to cm
        '''
        r, c = cm.shape

        if r != c:
            raise ValueError('Correlation matrix %s is not square' % cm)

        for i in xrange(r):
            for j in xrange(i+1):
                if cm.item(i, j) != cm.item(j, i):
                    raise ValueError('Correlation matrix %s is not symmetric' % cm)
        
        if not np.all(linalg.eigvalsh(cm) > 0):
            raise ValueError('Correlation matrix %s is not positive definite' % cm)

        self._corr_matrix = cm

    def create_sample(self, n_samples=1, time_step=1, *args):
        '''
        Generate a single path for each random variable (number of random variables, eg number of Brownian motions, is self.size). One path consists of n_samples steps
        Args:
            n_samples (int): the number of steps along a single path 
            time_step (float): the size of a mini time step. For Brownian motion, time_step is variance of the normal dist
        Returns: 
            ndarray: a 2D array (num rows: self.size, num cols: n_samples). Each row is one single path for one random variable. The steps are correlated by self._corr_matrix
        '''
        # try to see if a transform matrix C is already made for this time_step
        transform = self._ts_transforms.get(time_step)
        # if there is no matrix C yet then create one for this time_step
        if transform is None:
            transform = self._create_transform(time_step)
            self._ts_transforms[time_step] = transform
        
        # first create a set of independent paths for the group of variables
        # these iid samples must be standard normal before transforming
        # here iid is n_samples columns of Z, where Z is a column of std normal
        iid = super(CorrelatedSampleCreator, self).create_sample(n_samples, time_step=1)
        # matrix transform is C, and Y = C*Z so covar(Y) = C*C.T
        # matrix.A returns self as ndarray
        return (transform * matrix(iid)).A

    def _create_transform(self, time_step):
        '''
        From a time_step dt, calculate covar matrix and the transform matrix C where C * C.T = the target covar matrix
        Args:
            time_step: variance of each step of the Brownian motion
        Returns:
            matrix: the transformation matrix C
        '''
        # first create a diag matrix where the diag is sqrt(dt)
        ts = time_step ** 0.5
        I = matrix(ts*identity(self.size))

        # make the covar matrix based on dt and corr matrix 
        covar = I * self._corr_matrix * I

        # if Z is a column of std normal independent variables, then Y = C*Z 
        # then covar(Y) = C * C.T
        # here we return the transform matrix C
        return matrix(linalg.cholesky(covar))