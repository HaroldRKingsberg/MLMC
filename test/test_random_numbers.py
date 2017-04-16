import numpy
import unittest

from mlmc.src.random_numbers import IIDSampleCreator

class IIDSampleCreatorTestCase(unittest.TestCase):

    def test_dimensional_correctness(self):
        for sample_size in (3, 5):
            for n_samples in (2, 4):
                sut = IIDSampleCreator(sample_size)
                res = sut.create_sample(n_samples)
                self.assertEqual(res.shape, (sample_size, n_samples))

    def test_values_returned(self):
        sample_size = 5
        n_samples = 3
        distro = lambda size: numpy.array(range(size))
        ev = numpy.array([range(i, i+n_samples) for i in xrange(0, sample_size*n_samples, n_samples)])

        sut = IIDSampleCreator(sample_size, distro)
        res = sut.create_sample(n_samples)

        numpy.testing.assert_array_equal(res, ev)


if __name__ == '__main__':
    unittest.main()
