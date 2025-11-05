"""
Unit tests for prime classification functions
"""

import unittest
import numpy as np
import pandas as pd
from prime_classification import (
    is_prime, 
    generate_primes, 
    calculate_features,
    generate_non_primes,
    create_pca_transformer,
    perform_clustering
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
        
        # Check digit patterns
        self.assertEqual(features['digit_alternation'], 1)  # 1 != 3
        self.assertEqual(features['max_digit'], 3)
        self.assertEqual(features['min_digit'], 1)
    
    def test_features_have_correct_count(self):
        """Test that we generate the expected number of features"""
        features = calculate_features(123)
        
        # Should have 12 features (including number itself, prime added separately)
        # Basic properties: number, last_digit, sum_of_digits, num_digits, first_digit (5)
        # Digit patterns: digit_alternation, max_digit, min_digit, digit_range (4)
        # Digit operations: product_of_digits (1)
        # Variance: digit_variance, digit_std (2)
        # Total: 5 + 4 + 1 + 2 = 12
        self.assertEqual(len(features), 12)
    
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
    
    def test_pca_transformer_creation(self):
        """Test PCA transformer creation"""
        # Create a small dataset
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(100, 12))
        
        pca, scaler = create_pca_transformer(X_train, n_components=2)
        
        # Check that PCA and scaler are created
        self.assertIsNotNone(pca)
        self.assertIsNotNone(scaler)
        
        # Check PCA properties
        self.assertEqual(pca.n_components_, 2)
        self.assertEqual(len(pca.explained_variance_ratio_), 2)
        
        # Check that variance ratios sum to less than or equal to 1
        self.assertLessEqual(sum(pca.explained_variance_ratio_), 1.0)
    
    def test_pca_transformation(self):
        """Test that PCA transformation works correctly"""
        # Create a small dataset
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(100, 12))
        
        pca, scaler = create_pca_transformer(X_train, n_components=2)
        
        # Transform the data
        X_scaled = scaler.transform(X_train)
        X_pca = pca.transform(X_scaled)
        
        # Check transformed shape
        self.assertEqual(X_pca.shape, (100, 2))
    
    def test_clustering(self):
        """Test K-means clustering"""
        # Create a small dataset
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(100, 12))
        y_train = pd.Series([0, 1] * 50)
        
        pca, scaler = create_pca_transformer(X_train, n_components=2)
        kmeans, cluster_labels = perform_clustering(X_train, y_train, pca, scaler, n_clusters=3)
        
        # Check clustering results
        self.assertIsNotNone(kmeans)
        self.assertIsNotNone(cluster_labels)
        
        # Check cluster properties
        self.assertEqual(len(cluster_labels), 100)
        self.assertEqual(kmeans.n_clusters, 3)
        self.assertEqual(kmeans.cluster_centers_.shape, (3, 2))
        
        # Check that all cluster labels are valid
        self.assertTrue(all(label in [0, 1, 2] for label in cluster_labels))


if __name__ == '__main__':
    unittest.main()
