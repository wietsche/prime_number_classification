# Prime Number Classification

Machine Learning project for classifying prime numbers using various algorithms and feature engineering.

## Overview

This project generates a dataset of prime and non-prime numbers (ending in 1, 3, 7, or 9) up to 2^16 (65,536), extracts multiple features for each number, and trains various machine learning models to classify them. The project includes:

- Prime number generation with specific constraints
- Balanced dataset creation (equal primes and non-primes)
- Comprehensive feature engineering (28 features per number)
- Multiple ML algorithms with hyperparameter tuning
- Model evaluation and comparison
- Feature importance analysis
- Automated PDF report generation

## Features Calculated

For each number, the following features are computed:

1. **Basic Properties**: last digit, sum of digits, number of digits, first digit
2. **Divisibility Tests**: divisible by 2, 3, 5, 7, 11
3. **Modulo Operations**: mod 4, 6, 8, 9, 10
4. **Mathematical Properties**: square root, logarithm, even/odd flags
5. **Digit Patterns**: alternation, max/min digits, range, variance, standard deviation
6. **Factor Analysis**: approximate factor count
7. **Digit Operations**: product of digits

## Machine Learning Models

The following algorithms are trained with hyperparameter search:

1. **Random Forest Classifier**
2. **Gradient Boosting Classifier**
3. **Logistic Regression**
4. **Support Vector Machine (SVM)**
5. **Decision Tree Classifier**

Each model undergoes GridSearchCV to find optimal hyperparameters.

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
4. Train all models with hyperparameter search
5. Evaluate model performance
6. Generate feature importance rankings
7. Create a comprehensive PDF report

## Output Files

- `prime_dataset.csv`: Complete dataset with all features and labels
- `prime_classification_report.pdf`: Comprehensive analysis report including:
  - Dataset summary
  - Model performance comparisons
  - Best models for precision and recall
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
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
├── prime_dataset.csv           # Generated dataset (not in git)
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

## Results

The PDF report provides detailed insights into:
- Which features are most important for classification
- Comparative performance of different algorithms
- Optimal hyperparameters for each model
- Trade-offs between precision and recall

## License

This project is part of a machine learning research exercise.