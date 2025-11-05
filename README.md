# Prime Number Classification

Machine Learning project for classifying prime numbers using various algorithms and feature engineering, including PCA analysis and clustering.

## Overview

This project generates a dataset of prime and non-prime numbers (ending in 1, 3, 7, or 9) up to 2^16 (65,536), extracts digit patterns and basic properties for each number, and trains various machine learning models to classify them. The project includes:

- Prime number generation with specific constraints
- Balanced dataset creation (equal primes and non-primes)
- Feature engineering using digit patterns and basic properties (12 features per number)
- Multiple ML algorithms with hyperparameter tuning
- **PCA (Principal Component Analysis)** for dimensionality reduction
- Model evaluation on both original and PCA-transformed data
- **K-means clustering** analysis
- **2D visualizations** of primes vs non-primes in PCA space
- Feature importance analysis
- Automated PDF report generation with comprehensive analysis

## Features Calculated

For each number, the following features are computed:

1. **Basic Properties**: last digit, sum of digits, number of digits, first digit
2. **Digit Patterns**: alternation, max/min digits, range, variance, standard deviation
3. **Digit Operations**: product of digits

## Machine Learning Models

The following algorithms are trained with hyperparameter search:

1. **Random Forest Classifier**
2. **Gradient Boosting Classifier**
3. **Logistic Regression**
4. **Support Vector Machine (SVM)**
5. **Decision Tree Classifier**

Each model undergoes GridSearchCV to find optimal hyperparameters. Models are trained on both:
- **Original features** (12 dimensions)
- **PCA-transformed features** (2 dimensions)

## PCA Analysis

The project includes comprehensive PCA (Principal Component Analysis):

- **Dimensionality Reduction**: Reduces 12 features to 2 principal components
- **Variance Analysis**: Reports explained variance for each component
- **Model Performance**: Compares model performance on original vs PCA-transformed data
- **Transformer Persistence**: Saves PCA transformer for future use

## Clustering Analysis

K-means clustering is applied to the PCA-transformed data:

- **Number of Clusters**: 3 clusters (configurable)
- **2D Visualization**: Cluster overlay on PCA space
- **Distribution Analysis**: Reports cluster sizes and proportions

## Visualizations

The project generates multiple visualizations:

1. **Primes vs Non-primes in 2D PCA Space**: Shows the spread and separation of prime and non-prime numbers
2. **Clustering Overlay**: Displays K-means cluster assignments on PCA space
3. **Performance Comparison**: Original vs PCA-transformed model metrics
4. **Feature Importance**: Rankings for best-performing models

## Dataset Split

- **Training Set**: 70%
- **Testing Set**: 15%
- **Evaluation Set**: 15%

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the complete pipeline:

```bash
python prime_classification.py
```

This will:
1. Generate the prime and non-prime dataset
2. Calculate features for all numbers
3. Split the data into train/test/eval sets
4. Train all models with hyperparameter search on original features
5. **Perform PCA analysis and create PCA transformer**
6. **Train all models on PCA-transformed data**
7. **Perform K-means clustering on PCA space**
8. Evaluate model performance on both original and PCA-transformed data
9. Generate feature importance rankings
10. **Create 2D visualizations of primes vs non-primes**
11. **Overlay clustering information on visualizations**
12. Create a comprehensive PDF report with all analyses

## Output Files

- `prime_dataset.csv`: Complete dataset with all features and labels
- `pca_transformer.pkl`: Saved PCA transformer and scaler (for reuse)
- `prime_classification_report.pdf`: Comprehensive analysis report including:
  - Dataset summary
  - Model performance comparisons (original and PCA)
  - Best models for precision and recall
  - **2D PCA visualizations** of primes vs non-primes
  - **Clustering overlay** on PCA space
  - **Performance comparison** between original and PCA-transformed data
  - **PCA variance analysis**
  - Feature importance rankings

## Model Selection

The project identifies:
- **Best model for maximum precision**: Prioritizes reducing false positives
- **Best model for maximum recall**: Prioritizes finding all true positives

## Feature Importance

Feature importance is calculated and ranked for the best-performing models, helping identify which number properties are most predictive of primality.

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Project Structure

```
prime_number_classification/
├── prime_classification.py    # Main implementation
├── requirements.txt            # Python dependencies
├── test_prime_classification.py # Unit tests
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
├── prime_dataset.csv           # Generated dataset (not in git)
├── pca_transformer.pkl         # PCA transformer (not in git)
└── prime_classification_report.pdf  # Generated report (not in git)
```

## Methodology

1. **Prime Generation**: Uses trial division to identify primes
2. **Feature Engineering**: Computationally simple operations on integers
3. **Balancing**: Equal number of primes and non-primes
4. **Constraint**: All numbers end in 1, 3, 7, or 9
5. **Scaling**: Standard scaling applied where beneficial
6. **Cross-Validation**: 3-fold CV during hyperparameter search
7. **Metrics**: Precision, Recall, and F1-Score
8. **PCA**: Dimensionality reduction to 2 components for visualization and analysis
9. **Clustering**: K-means with 3 clusters on PCA-transformed data

## Results

The PDF report provides detailed insights into:
- Which features are most important for classification
- Comparative performance of different algorithms
- Optimal hyperparameters for each model
- Trade-offs between precision and recall
- **Impact of PCA dimensionality reduction** on model performance
- **Visual separation** of primes and non-primes in 2D space
- **Clustering patterns** in the feature space
- **Explained variance** by principal components

### Key Findings

- **PCA Analysis**: The first 2 principal components capture approximately 62% of the total variance
- **Model Performance**: Models trained on original features generally outperform PCA-transformed models, showing that the full feature space is valuable for this classification task
- **Visualization**: 2D PCA plots reveal the distribution and overlap patterns between prime and non-prime numbers
- **Clustering**: K-means clustering identifies natural groupings in the feature space that may correspond to different numerical properties

## License

This project is part of a machine learning research exercise.