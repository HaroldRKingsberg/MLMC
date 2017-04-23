import abc

from numpy import array, identity, linalg, matrix, random

class SampleCreator(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, size):
        self.size = size

    @abc.abstractmethod
    def create_sample(self, n_samples=1, *args):
        '''
        Return a numpy array of samples. Assume desired sample is of form (s1, s2, ...)
        Returned object will be of form [(s1_0, s1_1, s1_2, ...), (s2_0, ...), ...]
        '''


class IIDSampleCreator(SampleCreator):

    def __init__(self, size, scales=None, distro=random.normal):
        self.scales = scales or [1 for _ in xrange(size)]

        if len(self.scales) != size:
            raise ValueError('The number of scales does not match the size')

        self.distro = distro
        super(IIDSampleCreator, self).__init__(size)

    def create_sample(self, n_samples=1, *args):
        return array([
            self.distro(scale=scale, size=n_samples)
            for scale in self.scales
        ])


class CorrelatedSampleCreator(IIDSampleCreator):

    def __init__(self, corr_matrix, scales=None, distro=random.normal):
        self._set_corr_matrix(corr_matrix)
        size = len(corr_matrix)
        self._ts_transforms = {}

        super(CorrelatedSampleCreator, self).__init__(size=size,
                                                      scales=scales, 
                                                      distro=distro)

    def _set_corr_matrix(self, cm):
        r, c = cm.shape

        if r != c:
            raise ValueError('Correlation matrix %s is not square' % cm)

        for i in xrange(r):
            for j in xrange(i+1):
                if cm.item(i, j) != cm.item(j, i):
                    raise ValueError('Correlation matrix %s is not symmetric' % cm)

        self._corr_matrix = cm

    def create_sample(self, n_samples=1, time_step=1, *args):
        transform = self._ts_transforms.get(time_step)

        if transform is None:
            transform = self._create_transform(time_step)
            self._ts_transforms[time_step] = transform

        iid = super(CorrelatedSampleCreator, self).create_sample(n_samples)
        return (transform * matrix(iid)).A

    def _create_transform(self, time_step):
        ts = time_step ** 0.5
        I = matrix(ts*identity(self.size))
        covar = I * self._corr_matrix * I
        return matrix(linalg.cholesky(covar))

