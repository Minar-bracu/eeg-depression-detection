"""
Machine Learning Models Module

Trains and evaluates Random Forest and SVM classifiers for EEG-based depression detection
with cross-validation and performance metrics.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
from joblib import dump, load

from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


class EEGDepressionModelTrainer:
    """
    Trains and evaluates ML models for EEG depression detection.
    
    Parameters:
    -----------
    random_state : int, default 42
        Random seed for reproducibility
    cv_folds : int, default 5
        Number of cross-validation folds
    """
    
    def __init__(self, random_state=42, cv_folds=5):
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
    
    def create_models(self):
        """Initialize Random Forest and SVM models."""
        self.models['RandomForest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.models['SVM'] = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=self.random_state,
            class_weight='balanced'
        )
    
    def prepare_data(self, X, y):
        """
        Standardize features.
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix
        y : ndarray, shape (n_samples,)
            Binary labels
            
        Returns:
        --------
        X_scaled : ndarray
            Standardized feature matrix
        """
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled
    
    def cross_validate_model(self, model, X, y, model_name='Model'):
        """
        Perform stratified k-fold cross-validation.
        
        Parameters:
        -----------
        model : sklearn estimator
            Initialized model
        X : ndarray, shape (n_samples, n_features)
            Scaled feature matrix
        y : ndarray, shape (n_samples,)
            Binary labels
        model_name : str
            Name of model (for reporting)
            
        Returns:
        --------
        cv_results : dict
            Cross-validation results and metrics
        """
        cv_splitter = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        }
        
        # Cross-validate
        cv_results = cross_validate(
            model, X, y, cv=cv_splitter,
            scoring=scoring, return_train_score=True
        )
        
        # Compute mean and std
        results_summary = {}
        for metric in scoring.keys():
            test_scores = cv_results[f'test_{metric}']
            results_summary[metric] = {
                'mean': np.mean(test_scores),
                'std': np.std(test_scores),
                'scores': test_scores.tolist()
            }
        
        return results_summary
    
    def train_models(self, X, y):
        """
        Train all models on full dataset (after CV evaluation).
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            Scaled feature matrix
        y : ndarray, shape (n_samples,)
            Binary labels
            
        Returns:
        --------
        trained_models : dict
            Fitted model objects
        """
        trained_models = {}
        
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            model.fit(X, y)
            trained_models[model_name] = model
        
        return trained_models
    
    def get_feature_importance(self, model):
        """
        Get feature importance (for tree-based models).
        
        Parameters:
        -----------
        model : RandomForestClassifier
            Trained RF model
            
        Returns:
        --------
        importances : ndarray
            Feature importance scores
        """
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        return None
    
    def evaluate_pipeline(self, X, y):
        """
        Comprehensive evaluation: cross-validation + training.
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_features)
            Feature matrix
        y : ndarray, shape (n_samples,)
            Binary labels
            
        Returns:
        --------
        results : dict
            Evaluation results for all models
        """
        # Initialize models
        self.create_models()
        
        # Prepare data
        X_scaled = self.prepare_data(X, y)
        
        # Cross-validate each model
        all_results = {}
        for model_name, model in self.models.items():
            print(f"\n--- Cross-validating {model_name} ---")
            cv_results = self.cross_validate_model(model, X_scaled, y, model_name)
            all_results[model_name] = cv_results
            
            # Print summary
            print(f"Accuracy: {cv_results['accuracy']['mean']:.4f} ± {cv_results['accuracy']['std']:.4f}")
            print(f"F1-Score: {cv_results['f1']['mean']:.4f} ± {cv_results['f1']['std']:.4f}")
            print(f"ROC-AUC:  {cv_results['roc_auc']['mean']:.4f} ± {cv_results['roc_auc']['std']:.4f}")
        
        # Train final models
        print("\n--- Training final models ---")
        trained_models = self.train_models(X_scaled, y)
        
        self.results = {
            'cv_results': all_results,
            'trained_models': trained_models,
            'X_scaled': X_scaled,
            'y': y
        }
        
        return all_results
    
    def get_model_comparison_df(self):
        """
        Create comparison DataFrame for all models.
        
        Returns:
        --------
        comparison_df : pd.DataFrame
            Model comparison metrics
        """
        rows = []
        for model_name, results in self.results['cv_results'].items():
            row = {
                'Model': model_name,
                'Accuracy': f"{results['accuracy']['mean']:.4f} ± {results['accuracy']['std']:.4f}",
                'Precision': f"{results['precision']['mean']:.4f} ± {results['precision']['std']:.4f}",
                'Recall': f"{results['recall']['mean']:.4f} ± {results['recall']['std']:.4f}",
                'F1-Score': f"{results['f1']['mean']:.4f} ± {results['f1']['std']:.4f}",
                'ROC-AUC': f"{results['roc_auc']['mean']:.4f} ± {results['roc_auc']['std']:.4f}"
            }
            rows.append(row)
        
        return pd.DataFrame(rows)


def save_results(results, output_dir):
    """
    Save cross-validation results to JSON.
    
    Parameters:
    -----------
    results : dict
        CV results from trainer
    output_dir : str
        Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare results for JSON serialization
    json_results = {}
    for model_name, model_results in results.items():
        json_results[model_name] = {}
        for metric, values in model_results.items():
            json_results[model_name][metric] = {
                'mean': float(values['mean']),
                'std': float(values['std']),
                'scores': [float(s) for s in values['scores']]
            }
    
    # Save to JSON
    output_file = output_path / 'metrics.json'
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=4)
    
    print(f"Results saved to {output_file}")


def save_models(trainer, output_dir):
    """
    Save trained models to disk.
    
    Parameters:
    -----------
    trainer : EEGDepressionModelTrainer
        Trained trainer object
    output_dir : str
        Output directory
    """
    output_path = Path(output_dir) / 'models'
    output_path.mkdir(parents=True, exist_ok=True)
    
    for model_name, model in trainer.results['trained_models'].items():
        model_file = output_path / f'{model_name}.joblib'
        dump(model, model_file)
        print(f"Saved {model_name} to {model_file}")


if __name__ == "__main__":
    # Example usage
    print("Testing model training pipeline...")
    
    # Generate synthetic feature data
    np.random.seed(42)
    n_samples = 100
    n_features = 372  # 62 channels × 6 features per channel
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    # Initialize and train
    trainer = EEGDepressionModelTrainer(cv_folds=3)
    results = trainer.evaluate_pipeline(X, y)
    
    # Display results
    print("\n--- Model Comparison ---")
    print(trainer.get_model_comparison_df())
    
    print("\nModel training test complete!")
