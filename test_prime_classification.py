"""
Unit tests for prime classification functions
"""

import unittest
import numpy as np
from prime_classification import (
    is_prime, 
    generate_primes, 
    calculate_features,
    generate_non_primes
)


class TestPrimeClassification(unittest.TestCase):
    
    def test_is_prime(self):
        """Test prime number detection"""
        # Known primes
        self.assertTrue(is_prime(2))
        self.assertTrue(is_prime(3))
        self.assertTrue(is_prime(7))
        self.assertTrue(is_prime(11))
        self.assertTrue(is_prime(13))
        
        # Known non-primes
        self.assertFalse(is_prime(1))
        self.assertFalse(is_prime(4))
        self.assertFalse(is_prime(9))
        self.assertFalse(is_prime(15))
        self.assertFalse(is_prime(21))
    
    def test_generate_primes_ending_constraint(self):
        """Test that all generated primes end with 1,3,7,9"""
        primes = generate_primes(1000)
        
        for p in primes:
            last_digit = p % 10
            self.assertIn(last_digit, [1, 3, 7, 9], 
                         f"Prime {p} doesn't end with 1,3,7,9")
            self.assertTrue(is_prime(p), f"{p} is not prime")
    
    def test_generate_primes_count(self):
        """Test that we generate reasonable number of primes"""
        primes = generate_primes(100)
        # There should be several primes under 100 ending in 1,3,7,9
        self.assertGreater(len(primes), 10)
    
    def test_generate_non_primes(self):
        """Test non-prime generation"""
        primes = generate_primes(100)
        non_primes = generate_non_primes(primes, 100)
        
        # Should have equal counts
        self.assertEqual(len(primes), len(non_primes))
        
        # All should end with 1,3,7,9
        for n in non_primes:
            last_digit = n % 10
            self.assertIn(last_digit, [1, 3, 7, 9],
                         f"Non-prime {n} doesn't end with 1,3,7,9")
            self.assertFalse(is_prime(n), f"{n} is prime but should not be")
    
    def test_calculate_features(self):
        """Test feature calculation"""
        features = calculate_features(13)
        
        # Check basic features
        self.assertEqual(features['number'], 13)
        self.assertEqual(features['last_digit'], 3)
        self.assertEqual(features['num_digits'], 2)
        self.assertEqual(features['sum_of_digits'], 4)  # 1 + 3
        self.assertEqual(features['first_digit'], 1)
        
        # Check divisibility
        self.assertEqual(features['divisible_by_2'], 0)
        self.assertEqual(features['divisible_by_3'], 0)
        
        # Check is_odd
        self.assertEqual(features['is_odd'], 1)
        self.assertEqual(features['is_even'], 0)
    
    def test_features_have_correct_count(self):
        """Test that we generate the expected number of features"""
        features = calculate_features(123)
        
        # Should have 27 features (including number itself, prime added separately)
        self.assertEqual(len(features), 27)
    
    def test_edge_case_small_numbers(self):
        """Test with small numbers"""
        # Test with 1 (not prime)
        self.assertFalse(is_prime(1))
        
        # Test with 2 (prime but even)
        self.assertTrue(is_prime(2))
        
        # Features for small number
        features = calculate_features(3)
        self.assertEqual(features['num_digits'], 1)
        self.assertEqual(features['sum_of_digits'], 3)


if __name__ == '__main__':
    unittest.main()
