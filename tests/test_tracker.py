import numpy as np
import unittest
import scipy.stats as ss
from mlmc.option import StatTracker

class StatTrackerTestCase(unittest.TestCase):
    def test_stdev(self):
        n = 100
        rand_array = np.random.randn(n) + 100
        expected_stdev = np.std(rand_array)
        
        discount = 1        
        tracker = StatTracker(discount)
        for i in range(n):
            x = rand_array[i]
            tracker.add_sample(x)
        
        self.assertAlmostEqual(tracker.stdev, expected_stdev, 4)
        self.assertEqual(tracker.count, n)
    
    def test_var(self):
        n = 100
        rand_array = np.random.randn(n) + 100
        expected_var = np.std(rand_array)**2
        
        discount = 1        
        tracker = StatTracker(discount)
        for i in range(n):
            x = rand_array[i]
            tracker.add_sample(x)
        
        self.assertAlmostEqual(tracker.variance, expected_var, 4)
    
    def test_get_interval_length(self):
        n = 100
        rand_array = np.random.randn(n) + 100
        confidence_level = 0.95
        z_score = ss.norm.ppf(1 - (1 - confidence_level) / 2)
        interval_length = z_score * np.std(rand_array)
        
        discount = 1        
        tracker = StatTracker(discount)
        for i in range(n):
            x = rand_array[i]
            tracker.add_sample(x)
        
        self.assertAlmostEqual(tracker.get_interval_length(z_score), interval_length, 4)
    
if __name__ == '__main__':
    unittest.main()