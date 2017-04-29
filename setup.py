from setuptools import setup
import unittest


def get_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    return test_suite

setup(
    name='mlmc',
    version='.1',
    author='HMP',
    url='https://github.com/HaroldRKingsberg/MLMC',
    packages=['mlmc'],
    install_requires=['numpy'],
    test_suite='setup.get_test_suite'
)
