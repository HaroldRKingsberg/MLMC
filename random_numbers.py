import abc

from numpy import identity, linalg, matrix, random

class SampleCreator(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, size):
        self.size = size

    @abc.abstractmethod
    def create_sample(self, *args):
        ''' Return a numpy array of samples '''


class IIDSampleCreator(SampleCreator):

    def __init__(self, size, distro=random.normal):
        self.distro = distro
        super(IIDSampleCreator, self).__init__(size)

    def create_sample(self, *args):
        return self.distro(size=self.size)


class CorrelatedSampleCreator(SampleCreator):

    def __init__(self, size, corr_matrix):
        super(CorrelatedSampleCreator, self).__init__(size)
        self.corr_matrix = corr_matrix

        self._ts_transforms = {}

    @property
    def corr_matrix(self):
        return self._corr_matrix

    @corr_matrix.setter
    def corr_matrix(self, value):
        r, c = value.shape

        if r != c != self.size:
            raise ValueError('Correlation matrix %s is not square with sides %d' % (value, self.size))

        for i in xrange(r):
            for j in xrange(i+1):
                if value.item(i, j) != value.item(j, i):
                    raise ValueError('Correlation matrix %s is not symmetric' % value)

        self._corr_matrix = value

    def create_sample(self, time_step, *args):
        transform = self._ts_transforms.get(time_step)

        if transform is None:
            print("Whatever")
            transform = self._create_transform(time_step)
            self._ts_transforms[time_step] = transform

        sample = matrix(random.normal(size=self.size)).transpose()
        return (transform * sample).A1

    def _create_transform(self, time_step):
        ts = time_step ** 0.5
        I = matrix(ts*identity(self.size))
        covar = I*self.corr_matrix*I
        return matrix(linalg.cholesky(covar))

