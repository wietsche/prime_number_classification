"""
Prime Number Classification ML Project
Generates dataset, trains models, and analyzes feature importance
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import warnings
warnings.filterwarnings('ignore')

# Constants
LOGISTIC_REGRESSION_MAX_ITER = 1000  # Maximum iterations for logistic regression


def is_prime(n):
    """Check if a number is prime"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def generate_primes(max_val):
    """Generate list of prime numbers up to max_val ending in 1,3,7,9"""
    primes = []
    for n in range(2, max_val + 1):
        last_digit = n % 10
        if last_digit in [1, 3, 7, 9] and is_prime(n):
            primes.append(n)
    return primes


def generate_non_primes(primes, max_val):
    """Generate equal amount of non-prime numbers ending in 1,3,7,9"""
    non_primes = []
    n = 1
    target_count = len(primes)
    
    # First pass: collect all non-primes within max_val
    while n <= max_val:
        last_digit = n % 10
        if last_digit in [1, 3, 7, 9] and not is_prime(n) and n > 1:
            non_primes.append(n)
        n += 1
    
    # Sample to match exactly the target count
    np.random.seed(42)
    if len(non_primes) >= target_count:
        non_primes = list(np.random.choice(non_primes, size=target_count, replace=False))
    else:
        # If we don't have enough, use all available and sample with replacement
        non_primes = list(np.random.choice(non_primes, size=target_count, replace=True))
    
    return non_primes


def calculate_features(n):
    """Calculate digit patterns and basic properties for a number"""
    features = {}
    
    # Basic properties
    features['number'] = n
    features['last_digit'] = n % 10
    features['sum_of_digits'] = sum(int(d) for d in str(n))
    features['num_digits'] = len(str(n))
    features['first_digit'] = int(str(n)[0])
    
    # Digit patterns
    digits_str = str(n)
    features['digit_alternation'] = sum(1 for i in range(len(digits_str)-1) 
                                        if int(digits_str[i]) != int(digits_str[i+1]))
    features['max_digit'] = max(int(d) for d in digits_str)
    features['min_digit'] = min(int(d) for d in digits_str)
    features['digit_range'] = features['max_digit'] - features['min_digit']
    
    # Product of digits
    features['product_of_digits'] = np.prod([int(d) for d in digits_str])
    
    # Variance of digits
    digit_list = [int(d) for d in digits_str]
    features['digit_variance'] = np.var(digit_list)
    features['digit_std'] = np.std(digit_list)
    
    return features


def create_dataset(max_val):
    """Create complete dataset with features and labels"""
    print(f"Generating primes up to {max_val}...")
    primes = generate_primes(max_val)
    print(f"Found {len(primes)} primes")
    
    print("Generating non-primes...")
    non_primes = generate_non_primes(primes, max_val)
    print(f"Generated {len(non_primes)} non-primes")
    
    # Create dataset
    data = []
    
    print("Calculating features for primes...")
    for n in primes:
        features = calculate_features(n)
        features['prime'] = 1
        data.append(features)
    
    print("Calculating features for non-primes...")
    for n in non_primes:
        features = calculate_features(n)
        features['prime'] = 0
        data.append(features)
    
    df = pd.DataFrame(data)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


def split_dataset(df):
    """Split dataset into train, test, and eval sets"""
    # Separate features and target
    X = df.drop(['prime', 'number'], axis=1)
    y = df['prime']
    
    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Second split: 15% test, 15% eval
    X_test, X_eval, y_test, y_eval = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    return X_train, X_test, X_eval, y_train, y_test, y_eval


def create_pca_transformer(X_train, n_components=2):
    """Create and fit PCA transformer"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    pca = PCA(n_components=n_components)
    pca.fit(X_train_scaled)
    
    print(f"\nPCA Analysis:")
    print(f"Number of components: {n_components}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")
    
    return pca, scaler


def train_models(X_train, y_train, X_test, y_test):
    """Train multiple ML models with hyperparameter search"""
    results = {}
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models and their hyperparameter grids
    models = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=LOGISTIC_REGRESSION_MAX_ITER),
            'params': {
                'C': [0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'saga']
            }
        },
        'SVM': {
            'model': SVC(random_state=42),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
    }
    
    print("\nTraining models with hyperparameter search...")
    
    for name, config in models.items():
        print(f"\nTraining {name}...")
        
        # Use scaled data for models that benefit from it
        if name in ['Logistic Regression', 'SVM']:
            X_tr, X_te = X_train_scaled, X_test_scaled
        else:
            X_tr, X_te = X_train, X_test
        
        # Grid search
        grid_search = GridSearchCV(
            config['model'], 
            config['params'], 
            cv=3, 
            scoring='f1',
            n_jobs=-1
        )
        grid_search.fit(X_tr, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        
        # Predictions
        y_pred = best_model.predict(X_te)
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'scaler': scaler if name in ['Logistic Regression', 'SVM'] else None,
            'predictions': y_pred
        }
        
        print(f"{name} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return results


def find_best_models(results):
    """Find best models for precision and recall"""
    best_precision = max(results.items(), key=lambda x: x[1]['precision'])
    best_recall = max(results.items(), key=lambda x: x[1]['recall'])
    
    print(f"\n{'='*60}")
    print("BEST MODELS:")
    print(f"{'='*60}")
    print(f"Best Precision: {best_precision[0]} ({best_precision[1]['precision']:.4f})")
    print(f"Best Recall: {best_recall[0]} ({best_recall[1]['recall']:.4f})")
    print(f"{'='*60}")
    
    return best_precision, best_recall


def train_models_pca(X_train, y_train, X_test, y_test, pca, scaler):
    """Train models on PCA-transformed data"""
    results_pca = {}
    
    # Transform data using PCA
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_pca = pca.transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Define models and their hyperparameter grids
    models = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=LOGISTIC_REGRESSION_MAX_ITER),
            'params': {
                'C': [0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'saga']
            }
        },
        'SVM': {
            'model': SVC(random_state=42),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
    }
    
    print("\nTraining models on PCA-transformed data...")
    
    for name, config in models.items():
        print(f"\nTraining {name} on PCA data...")
        
        # Grid search
        grid_search = GridSearchCV(
            config['model'], 
            config['params'], 
            cv=3, 
            scoring='f1',
            n_jobs=-1
        )
        grid_search.fit(X_train_pca, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        
        # Predictions
        y_pred = best_model.predict(X_test_pca)
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results_pca[name] = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred
        }
        
        print(f"{name} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return results_pca


def perform_clustering(X_train, y_train, pca, scaler, n_clusters=3):
    """Perform K-means clustering on PCA-transformed data"""
    X_train_scaled = scaler.transform(X_train)
    X_train_pca = pca.transform(X_train_scaled)
    
    # Only use first 2 components for visualization
    X_train_pca_2d = X_train_pca[:, :2]
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_train_pca_2d)
    
    print(f"\nClustering Analysis:")
    print(f"Number of clusters: {n_clusters}")
    print(f"Cluster centers shape: {kmeans.cluster_centers_.shape}")
    
    return kmeans, cluster_labels


def calculate_feature_importance(model, feature_names, model_name):
    """Calculate feature importance for a model"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return None
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    return importance_df


def visualize_pca_2d(X_train, y_train, pca, scaler, kmeans, cluster_labels):
    """Create 2D visualization of PCA components with primes vs non-primes and clusters"""
    X_train_scaled = scaler.transform(X_train)
    X_train_pca = pca.transform(X_train_scaled)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Primes vs Non-primes
    ax1 = axes[0]
    primes_mask = y_train == 1
    non_primes_mask = y_train == 0
    
    ax1.scatter(X_train_pca[non_primes_mask, 0], X_train_pca[non_primes_mask, 1], 
               c='blue', alpha=0.5, s=20, label='Non-primes', edgecolors='none')
    ax1.scatter(X_train_pca[primes_mask, 0], X_train_pca[primes_mask, 1], 
               c='red', alpha=0.5, s=20, label='Primes', edgecolors='none')
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax1.set_title('Primes vs Non-primes in 2D PCA Space')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cluster overlay
    ax2 = axes[1]
    scatter = ax2.scatter(X_train_pca[:, 0], X_train_pca[:, 1], 
                         c=cluster_labels, cmap='viridis', alpha=0.5, s=20, edgecolors='none')
    
    # Plot cluster centers
    centers_pca = kmeans.cluster_centers_
    ax2.scatter(centers_pca[:, 0], centers_pca[:, 1], 
               c='red', marker='X', s=200, edgecolors='black', linewidth=2, 
               label='Cluster Centers')
    
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax2.set_title('K-means Clustering Overlay on PCA Space')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar for clusters
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Cluster Label')
    
    plt.tight_layout()
    return fig


def generate_pdf_report(df, results, best_precision, best_recall, X_train, 
                        results_pca=None, pca=None, scaler_pca=None, 
                        kmeans=None, cluster_labels=None, y_train=None):
    """Generate comprehensive PDF report"""
    print("\nGenerating PDF report...")
    
    pdf_path = 'prime_classification_report.pdf'
    
    with PdfPages(pdf_path) as pdf:
        # Page 1: Title and Summary
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.9, 'Prime Number Classification', 
                ha='center', va='top', fontsize=24, weight='bold')
        fig.text(0.5, 0.85, 'Machine Learning Analysis Report', 
                ha='center', va='top', fontsize=16)
        
        # Dataset summary
        summary_text = f"""
Dataset Summary:
- Total samples: {len(df)}
- Prime numbers: {sum(df['prime'] == 1)}
- Non-prime numbers: {sum(df['prime'] == 0)}
- Number of features: {len(X_train.columns)}
- Maximum value: {df['number'].max()}

Dataset Split:
- Training: 70%
- Testing: 15%
- Evaluation: 15%

Models Trained:
"""
        for name in results.keys():
            summary_text += f"- {name}\n"
        
        if results_pca:
            summary_text += f"""
PCA Analysis:
- PCA components: {pca.n_components_}
- Total variance explained: {sum(pca.explained_variance_ratio_):.4f}
- Models trained on PCA data: {len(results_pca)}
"""
        
        fig.text(0.1, 0.7, summary_text, ha='left', va='top', fontsize=11, 
                family='monospace')
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2: Model Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Model Performance Comparison', fontsize=16, weight='bold')
        
        model_names = list(results.keys())
        precisions = [results[name]['precision'] for name in model_names]
        recalls = [results[name]['recall'] for name in model_names]
        f1_scores = [results[name]['f1'] for name in model_names]
        
        # Precision bar chart
        axes[0, 0].bar(range(len(model_names)), precisions, color='skyblue')
        axes[0, 0].set_xticks(range(len(model_names)))
        axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].set_title('Precision by Model')
        axes[0, 0].set_ylim([0, 1])
        
        # Recall bar chart
        axes[0, 1].bar(range(len(model_names)), recalls, color='lightcoral')
        axes[0, 1].set_xticks(range(len(model_names)))
        axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].set_title('Recall by Model')
        axes[0, 1].set_ylim([0, 1])
        
        # F1 Score bar chart
        axes[1, 0].bar(range(len(model_names)), f1_scores, color='lightgreen')
        axes[1, 0].set_xticks(range(len(model_names)))
        axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('F1 Score by Model')
        axes[1, 0].set_ylim([0, 1])
        
        # Combined comparison
        x = np.arange(len(model_names))
        width = 0.25
        axes[1, 1].bar(x - width, precisions, width, label='Precision', color='skyblue')
        axes[1, 1].bar(x, recalls, width, label='Recall', color='lightcoral')
        axes[1, 1].bar(x + width, f1_scores, width, label='F1', color='lightgreen')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('All Metrics Comparison')
        axes[1, 1].legend()
        axes[1, 1].set_ylim([0, 1])
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 3: Best Models Summary
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.95, 'Best Models Analysis', 
                ha='center', va='top', fontsize=18, weight='bold')
        
        best_models_text = f"""
BEST MODEL FOR PRECISION:
Model: {best_precision[0]}
Precision: {best_precision[1]['precision']:.4f}
Recall: {best_precision[1]['recall']:.4f}
F1 Score: {best_precision[1]['f1']:.4f}

Best Hyperparameters:
"""
        for param, value in best_precision[1]['best_params'].items():
            best_models_text += f"  {param}: {value}\n"
        
        best_models_text += f"""

BEST MODEL FOR RECALL:
Model: {best_recall[0]}
Precision: {best_recall[1]['precision']:.4f}
Recall: {best_recall[1]['recall']:.4f}
F1 Score: {best_recall[1]['f1']:.4f}

Best Hyperparameters:
"""
        for param, value in best_recall[1]['best_params'].items():
            best_models_text += f"  {param}: {value}\n"
        
        fig.text(0.1, 0.85, best_models_text, ha='left', va='top', 
                fontsize=11, family='monospace')
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Add PCA visualization page if available
        if results_pca and pca and scaler_pca and kmeans and cluster_labels is not None and y_train is not None:
            # Page: PCA 2D Visualization
            fig = visualize_pca_2d(X_train, y_train, pca, scaler_pca, kmeans, cluster_labels)
            fig.suptitle('PCA Analysis and Clustering Visualization', fontsize=16, weight='bold', y=0.98)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page: PCA Performance Comparison
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            fig.suptitle('Original vs PCA-Transformed Model Performance', fontsize=16, weight='bold')
            
            model_names = list(results.keys())
            
            # Precision comparison
            precisions_orig = [results[name]['precision'] for name in model_names]
            precisions_pca = [results_pca[name]['precision'] for name in model_names]
            
            x = np.arange(len(model_names))
            width = 0.35
            axes[0, 0].bar(x - width/2, precisions_orig, width, label='Original', color='skyblue')
            axes[0, 0].bar(x + width/2, precisions_pca, width, label='PCA', color='orange')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
            axes[0, 0].set_ylabel('Precision')
            axes[0, 0].set_title('Precision: Original vs PCA')
            axes[0, 0].legend()
            axes[0, 0].set_ylim([0, 1])
            
            # Recall comparison
            recalls_orig = [results[name]['recall'] for name in model_names]
            recalls_pca = [results_pca[name]['recall'] for name in model_names]
            
            axes[0, 1].bar(x - width/2, recalls_orig, width, label='Original', color='lightcoral')
            axes[0, 1].bar(x + width/2, recalls_pca, width, label='PCA', color='orange')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
            axes[0, 1].set_ylabel('Recall')
            axes[0, 1].set_title('Recall: Original vs PCA')
            axes[0, 1].legend()
            axes[0, 1].set_ylim([0, 1])
            
            # F1 Score comparison
            f1_orig = [results[name]['f1'] for name in model_names]
            f1_pca = [results_pca[name]['f1'] for name in model_names]
            
            axes[1, 0].bar(x - width/2, f1_orig, width, label='Original', color='lightgreen')
            axes[1, 0].bar(x + width/2, f1_pca, width, label='PCA', color='orange')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].set_title('F1 Score: Original vs PCA')
            axes[1, 0].legend()
            axes[1, 0].set_ylim([0, 1])
            
            # Performance difference
            precision_diff = [results_pca[name]['precision'] - results[name]['precision'] for name in model_names]
            recall_diff = [results_pca[name]['recall'] - results[name]['recall'] for name in model_names]
            f1_diff = [results_pca[name]['f1'] - results[name]['f1'] for name in model_names]
            
            width = 0.25
            axes[1, 1].bar(x - width, precision_diff, width, label='Precision Δ', color='skyblue')
            axes[1, 1].bar(x, recall_diff, width, label='Recall Δ', color='lightcoral')
            axes[1, 1].bar(x + width, f1_diff, width, label='F1 Δ', color='lightgreen')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
            axes[1, 1].set_ylabel('Performance Difference (PCA - Original)')
            axes[1, 1].set_title('Performance Change with PCA')
            axes[1, 1].legend()
            axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page: PCA Analysis Details
            fig = plt.figure(figsize=(11, 8.5))
            fig.text(0.5, 0.95, 'PCA and Clustering Analysis Details', 
                    ha='center', va='top', fontsize=18, weight='bold')
            
            pca_text = f"""
PCA Analysis:
- Number of components: {pca.n_components_}
- Total variance explained: {sum(pca.explained_variance_ratio_):.4f}
- PC1 variance: {pca.explained_variance_ratio_[0]:.4f}
- PC2 variance: {pca.explained_variance_ratio_[1]:.4f}

K-means Clustering:
- Number of clusters: {kmeans.n_clusters}
- Inertia: {kmeans.inertia_:.2f}

Cluster Distribution:
"""
            unique, counts = np.unique(cluster_labels, return_counts=True)
            for cluster_id, count in zip(unique, counts):
                pca_text += f"- Cluster {cluster_id}: {count} samples ({count/len(cluster_labels)*100:.1f}%)\n"
            
            pca_text += f"""

Performance Summary (Original vs PCA):
"""
            for name in model_names:
                pca_text += f"\n{name}:\n"
                pca_text += f"  Original - P: {results[name]['precision']:.4f}, R: {results[name]['recall']:.4f}, F1: {results[name]['f1']:.4f}\n"
                pca_text += f"  PCA      - P: {results_pca[name]['precision']:.4f}, R: {results_pca[name]['recall']:.4f}, F1: {results_pca[name]['f1']:.4f}\n"
            
            fig.text(0.1, 0.85, pca_text, ha='left', va='top', 
                    fontsize=10, family='monospace')
            
            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Pages 4+: Feature Importance for best models
        for best_model_info, metric_name in [(best_precision, 'Precision'), 
                                               (best_recall, 'Recall')]:
            model_name = best_model_info[0]
            model = best_model_info[1]['model']
            
            importance_df = calculate_feature_importance(
                model, X_train.columns, model_name
            )
            
            if importance_df is not None:
                # Feature importance visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 8.5))
                fig.suptitle(f'Feature Importance - {model_name} (Best {metric_name})', 
                           fontsize=14, weight='bold')
                
                # Top 15 features bar chart
                top_features = importance_df.head(15)
                ax1.barh(range(len(top_features)), top_features['importance'])
                ax1.set_yticks(range(len(top_features)))
                ax1.set_yticklabels(top_features['feature'])
                ax1.invert_yaxis()
                ax1.set_xlabel('Importance')
                ax1.set_title('Top 15 Features')
                
                # All features text
                importance_text = "Feature Importance Rankings:\n\n"
                for idx, row in importance_df.iterrows():
                    importance_text += f"{row['feature']:25s}: {row['importance']:.6f}\n"
                
                ax2.text(0.05, 0.95, importance_text, ha='left', va='top', 
                        fontsize=8, family='monospace', transform=ax2.transAxes)
                ax2.axis('off')
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
    
    print(f"PDF report saved as: {pdf_path}")
    return pdf_path


def main():
    """Main execution function"""
    print("="*60)
    print("PRIME NUMBER CLASSIFICATION ML PROJECT")
    print("="*60)
    
    # Set parameters
    x = 16
    max_val = 2**x
    
    print(f"\nGenerating dataset for numbers up to 2^{x} = {max_val}")
    
    # Create dataset
    df = create_dataset(max_val)
    
    # Save dataset
    df.to_csv('prime_dataset.csv', index=False)
    print(f"\nDataset saved to: prime_dataset.csv")
    print(f"Dataset shape: {df.shape}")
    
    # Split dataset
    X_train, X_test, X_eval, y_train, y_test, y_eval = split_dataset(df)
    
    print(f"\nDataset splits:")
    print(f"Training: {len(X_train)} samples")
    print(f"Testing: {len(X_test)} samples")
    print(f"Evaluation: {len(X_eval)} samples")
    
    # Train models
    results = train_models(X_train, y_train, X_test, y_test)
    
    # Find best models
    best_precision, best_recall = find_best_models(results)
    
    # PCA Analysis
    print(f"\n{'='*60}")
    print("PCA ANALYSIS:")
    print(f"{'='*60}")
    
    pca, scaler_pca = create_pca_transformer(X_train, n_components=2)
    
    # Save PCA transformer
    with open('pca_transformer.pkl', 'wb') as f:
        pickle.dump({'pca': pca, 'scaler': scaler_pca}, f)
    print("PCA transformer saved to: pca_transformer.pkl")
    
    # Train models on PCA-transformed data
    results_pca = train_models_pca(X_train, y_train, X_test, y_test, pca, scaler_pca)
    
    # Perform clustering
    print(f"\n{'='*60}")
    print("CLUSTERING ANALYSIS:")
    print(f"{'='*60}")
    
    kmeans, cluster_labels = perform_clustering(X_train, y_train, pca, scaler_pca, n_clusters=3)
    
    # Evaluate on evaluation set
    print(f"\n{'='*60}")
    print("EVALUATION SET PERFORMANCE (ORIGINAL FEATURES):")
    print(f"{'='*60}")
    
    for name, result in results.items():
        model = result['model']
        scaler = result['scaler']
        
        if scaler:
            X_eval_transformed = scaler.transform(X_eval)
        else:
            X_eval_transformed = X_eval
        
        y_pred_eval = model.predict(X_eval_transformed)
        precision_eval = precision_score(y_eval, y_pred_eval)
        recall_eval = recall_score(y_eval, y_pred_eval)
        f1_eval = f1_score(y_eval, y_pred_eval)
        
        print(f"{name:25s} - P: {precision_eval:.4f}, R: {recall_eval:.4f}, F1: {f1_eval:.4f}")
    
    # Evaluate on evaluation set with PCA
    print(f"\n{'='*60}")
    print("EVALUATION SET PERFORMANCE (PCA-TRANSFORMED):")
    print(f"{'='*60}")
    
    X_eval_scaled = scaler_pca.transform(X_eval)
    X_eval_pca = pca.transform(X_eval_scaled)
    
    for name, result in results_pca.items():
        model = result['model']
        
        y_pred_eval = model.predict(X_eval_pca)
        precision_eval = precision_score(y_eval, y_pred_eval)
        recall_eval = recall_score(y_eval, y_pred_eval)
        f1_eval = f1_score(y_eval, y_pred_eval)
        
        print(f"{name:25s} - P: {precision_eval:.4f}, R: {recall_eval:.4f}, F1: {f1_eval:.4f}")
    
    # Generate PDF report
    pdf_path = generate_pdf_report(df, results, best_precision, best_recall, X_train,
                                   results_pca=results_pca, pca=pca, scaler_pca=scaler_pca,
                                   kmeans=kmeans, cluster_labels=cluster_labels, y_train=y_train)
    
    print(f"\n{'='*60}")
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"Dataset: prime_dataset.csv")
    print(f"PCA Transformer: pca_transformer.pkl")
    print(f"Report: {pdf_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
