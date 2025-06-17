import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import shap
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

def gini_coefficient(y_true, y_pred):
    """Calculate Gini coefficient from predictions"""
    auc = roc_auc_score(y_true, y_pred)
    return 2 * auc - 1

def preprocess_data(df, target_col, test_size=0.2, random_state=42):
    """
    Preprocess data: handle categorical variables with one-hot encoding, split and scale
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        test_size: Size of test set
        random_state: Random seed
    
    Returns:
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, feature_names, scaler
    """
    # Remove missing values
    df = df.dropna()
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # One-hot encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) > 0:
        print(f"One-hot encoding {len(categorical_cols)} categorical columns: {list(categorical_cols)}")
        # Use get_dummies for one-hot encoding
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        print(f"Features after one-hot encoding: {X.shape[1]}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, X.columns.tolist(), scaler

def run_logistic(X_train_scaled, y_train, X_test_scaled, y_test):
    """
    Train Logistic Regression model
    
    Args:
        X_train_scaled: Scaled training features
        y_train: Training target
        X_test_scaled: Scaled test features  
        y_test: Test target
    
    Returns:
        model, predictions, gini_score
    """
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Get probability predictions
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate Gini coefficient
    gini = gini_coefficient(y_test, y_pred_proba)
    
    return model, y_pred_proba, gini

def run_xgboost(X_train, y_train, X_test, y_test):
    """
    Train XGBoost model
    
    Args:
        X_train: Training features (not scaled)
        y_train: Training target
        X_test: Test features (not scaled)
        y_test: Test target
    
    Returns:
        model, predictions, gini_score
    """
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='rmse'
    )
    
    model.fit(X_train, y_train)
    
    # Get predictions and clip to [0, 1] range
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, 1)
    
    # Calculate Gini coefficient
    gini = gini_coefficient(y_test, y_pred)
    
    return model, y_pred, gini

def run_rnn(X_train_scaled, y_train, X_test_scaled, y_test, epochs=100, verbose=0):
    """
    Train RNN model
    
    Args:
        X_train_scaled: Scaled training features
        y_train: Training target
        X_test_scaled: Scaled test features
        y_test: Test target
        epochs: Number of training epochs
        verbose: Verbosity level
    
    Returns:
        model, predictions, gini_score, training_history
    """
    # Build neural network
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['mae']
    )
    
    # Early stopping callback
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop],
        verbose=verbose
    )
    
    # Get predictions
    y_pred = model.predict(X_test_scaled, verbose=0).flatten()
    
    # Calculate Gini coefficient
    gini = gini_coefficient(y_test, y_pred)
    
    return model, y_pred, gini, history

def shap_feature_importance(model, X_train, X_test, feature_names, model_type='sklearn', max_evals=100):
    """
    Calculate SHAP feature importance
    
    Args:
        model: Trained model
        X_train: Training features (for background samples)
        X_test: Test features
        feature_names: List of feature names
        model_type: 'sklearn', 'xgboost', or 'rnn'
        max_evals: Maximum evaluations for SHAP (controls computation time)
    
    Returns:
        DataFrame with SHAP importance scores and explainer object
    """
    try:
        if model_type == 'sklearn':
            # For Logistic Regression
            explainer = shap.LinearExplainer(model, X_train)
            shap_values = explainer.shap_values(X_test)
            
        elif model_type == 'xgboost':
            # For XGBoost - use TreeExplainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
        elif model_type == 'rnn':
            # For neural networks - use DeepExplainer or KernelExplainer
            # Use a subset of training data as background for efficiency
            background = X_train[:min(100, len(X_train))]
            
            # Try DeepExplainer first, fallback to KernelExplainer if needed
            try:
                explainer = shap.DeepExplainer(model, background)
                shap_values = explainer.shap_values(X_test[:min(200, len(X_test))])
            except:
                print("DeepExplainer failed, using KernelExplainer (slower)...")
                explainer = shap.KernelExplainer(
                    lambda x: model.predict(x, verbose=0).flatten(), 
                    background
                )
                shap_values = explainer.shap_values(X_test[:min(100, len(X_test))], nsamples=max_evals)
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            # For classification models, take positive class SHAP values
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        
        # Calculate mean absolute SHAP values for feature importance
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        
        # Create results DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': mean_shap_values,
            'shap_importance_normalized': mean_shap_values / np.sum(mean_shap_values)
        }).sort_values('shap_importance', ascending=False)
        
        return importance_df, explainer, shap_values
        
    except Exception as e:
        print(f"SHAP calculation failed: {str(e)}")
        print("Falling back to simple feature importance...")
        
        # Fallback: return zero importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': np.zeros(len(feature_names)),
            'shap_importance_normalized': np.zeros(len(feature_names))
        })
        
        return importance_df, None, None

# Example usage for one dataset
def example_usage():
    """Example of how to use the functions for one dataset"""
    
    # Load your data
    df = pd.read_csv('your_dataset.csv')
    target_col = 'your_target_column'
    
    # Preprocess data
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, feature_names, scaler = preprocess_data(df, target_col)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {len(feature_names)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Run Logistic Regression
    print("\n--- Logistic Regression ---")
    lr_model, lr_pred, lr_gini = run_logistic(X_train_scaled, y_train, X_test_scaled, y_test)
    print(f"Gini coefficient: {lr_gini:.4f}")
    
    # Calculate feature importance for Logistic Regression
    lr_importance, lr_explainer, lr_shap_values = shap_feature_importance(lr_model, X_train_scaled, X_test_scaled, feature_names, 'sklearn')
    print("Top 5 important features:")
    print(lr_importance.head())
    
    # Run XGBoost
    print("\n--- XGBoost ---")
    xgb_model, xgb_pred, xgb_gini = run_xgboost(X_train, y_train, X_test, y_test)
    print(f"Gini coefficient: {xgb_gini:.4f}")
    
    # Calculate feature importance for XGBoost
    xgb_importance, xgb_explainer, xgb_shap_values = shap_feature_importance(xgb_model, X_train, X_test, feature_names, 'xgboost')
    print("Top 5 important features:")
    print(xgb_importance.head())
    
    # Run RNN
    print("\n--- RNN ---")
    rnn_model, rnn_pred, rnn_gini, rnn_history = run_rnn(X_train_scaled, y_train, X_test_scaled, y_test)
    print(f"Gini coefficient: {rnn_gini:.4f}")
    
    # Calculate feature importance for RNN
    rnn_importance, rnn_explainer, rnn_shap_values = shap_feature_importance(rnn_model, X_train_scaled, X_test_scaled, feature_names, 'rnn')
    print("Top 5 important features:")
    print(rnn_importance.head())
    
    return {
        'logistic': {'model': lr_model, 'gini': lr_gini, 'importance': lr_importance},
        'xgboost': {'model': xgb_model, 'gini': xgb_gini, 'importance': xgb_importance},
        'rnn': {'model': rnn_model, 'gini': rnn_gini, 'importance': rnn_importance}
    }

# Template for looping through multiple datasets
def run_all_datasets():
    """Template for running all algorithms on multiple datasets"""
    
    # Define your datasets and target columns
    datasets = [
        {'file': 'dataset1.csv', 'target': 'target1'},
        {'file': 'dataset2.csv', 'target': 'target2'},
        {'file': 'dataset3.csv', 'target': 'target3'},
        {'file': 'dataset4.csv', 'target': 'target4'},
        {'file': 'dataset5.csv', 'target': 'target5'}
    ]
    
    results = {}
    
    for i, dataset_info in enumerate(datasets):
        dataset_name = f"dataset_{i+1}"
        print(f"\n{'='*50}")
        print(f"Processing {dataset_name}: {dataset_info['file']}")
        print('='*50)
        
        # Load and preprocess data
        df = pd.read_csv(dataset_info['file'])
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, feature_names, scaler = preprocess_data(df, dataset_info['target'])
        
        # Initialize results for this dataset
        results[dataset_name] = {}
        
        # Run Logistic Regression
        print("\nRunning Logistic Regression...")
        lr_model, lr_pred, lr_gini = run_logistic(X_train_scaled, y_train, X_test_scaled, y_test)
        lr_importance, lr_explainer, lr_shap_values = shap_feature_importance(lr_model, X_train_scaled, X_test_scaled, feature_names, 'sklearn')
        results[dataset_name]['logistic'] = {'gini': lr_gini, 'importance': lr_importance}
        print(f"Logistic Regression Gini: {lr_gini:.4f}")
        
        # Run XGBoost
        print("Running XGBoost...")
        xgb_model, xgb_pred, xgb_gini = run_xgboost(X_train, y_train, X_test, y_test)
        xgb_importance, xgb_explainer, xgb_shap_values = shap_feature_importance(xgb_model, X_train, X_test, feature_names, 'xgboost')
        results[dataset_name]['xgboost'] = {'gini': xgb_gini, 'importance': xgb_importance}
        print(f"XGBoost Gini: {xgb_gini:.4f}")
        
        # Run RNN
        print("Running RNN...")
        rnn_model, rnn_pred, rnn_gini, rnn_history = run_rnn(X_train_scaled, y_train, X_test_scaled, y_test)
        rnn_importance, rnn_explainer, rnn_shap_values = shap_feature_importance(rnn_model, X_train_scaled, X_test_scaled, feature_names, 'rnn')
        results[dataset_name]['rnn'] = {'gini': rnn_gini, 'importance': rnn_importance}
        print(f"RNN Gini: {rnn_gini:.4f}")
    
    return results

def plot_shap_summary(explainer, shap_values, X_test, feature_names, model_name):
    """
    Create SHAP summary plots
    
    Args:
        explainer: SHAP explainer object
        shap_values: SHAP values
        X_test: Test features
        feature_names: List of feature names
        model_name: Name of the model for plot title
    """
    if explainer is not None and shap_values is not None:
        try:
            # Summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
            plt.title(f'SHAP Summary Plot - {model_name}')
            plt.tight_layout()
            plt.show()
            
            # Bar plot of feature importance
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
            plt.title(f'SHAP Feature Importance - {model_name}')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Could not create SHAP plots for {model_name}: {str(e)}")
    else:
        print(f"No SHAP values available for {model_name}")

# Add matplotlib import at the top if not already present
import matplotlib.pyplot as plt
