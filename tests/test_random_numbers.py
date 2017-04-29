import itertools
import numpy
import unittest

from mlmc.random_numbers import IIDSampleCreator, CorrelatedSampleCreator

class IIDSampleCreatorTestCase(unittest.TestCase):

    def test_dimensional_correctness(self):
        for sample_size in (3, 5):
            for n_samples in (2, 4):
                sut = IIDSampleCreator(sample_size)
                res = sut.create_sample(n_samples)
                self.assertEqual(res.shape, (sample_size, n_samples))

    def test_size_attribute(self):
        sample_size = 4
        sut = IIDSampleCreator(sample_size)
        self.assertEqual(sut.size, sample_size)

    def test_values_returned(self):
        sample_size = 5
        n_samples = 3
        c = itertools.count()
        distro = lambda scale, size: numpy.array([next(c) for _ in xrange(size)])
        ev = numpy.array([range(i, i+n_samples) for i in xrange(0, sample_size*n_samples, n_samples)])

        sut = IIDSampleCreator(sample_size, distro=distro)
        res = sut.create_sample(n_samples)

        numpy.testing.assert_array_equal(res, ev)

    def test_scales_work(self):
        for s in [1, 2, 3]:
            sut = IIDSampleCreator(1)
            sample = sut.create_sample(n_samples=100000, time_step=s)[0]

            self.assertAlmostEqual(numpy.std(sample), s, 2)


class CorrelatedSampleCreatorTestCase(unittest.TestCase):

    def test_non_square_matrix_not_accepted(self):
        bad_corr = numpy.array([[1,2]])

        with self.assertRaises(ValueError):
            CorrelatedSampleCreator(bad_corr)

    def test_non_symmetric_matrix_not_accepted(self):
        bad_corr = numpy.array([[1, 1], [0, 1]])

        with self.assertRaises(ValueError):
            CorrelatedSampleCreator(bad_corr)

    def test_size_attribute(self):
        sample_size = 4
        corr = numpy.identity(sample_size)
        sut = CorrelatedSampleCreator(corr)

        self.assertEqual(sut.size, sample_size)

    def test_normalize_figures(self):
        sample_size = 5

        corr = numpy.identity(sample_size)

        for s in xrange(1, 5):
            c = itertools.count()
            distro = lambda scale, size: numpy.array([next(c) for _ in xrange(size)])

            sut = CorrelatedSampleCreator(corr, distro=distro)

            step = s ** 2
            res = sut.create_sample(n_samples=1, time_step=step)
            ev = numpy.array([[s*i] for i in xrange(sample_size)])
            numpy.testing.assert_array_equal(res, ev)


if __name__ == '__main__':
    unittest.main()
