import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
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
    Preprocess data: handle categorical variables, split and scale
    
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
    
    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
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

def permutation_feature_importance(model, X_test, y_test, feature_names, model_type='sklearn', n_repeats=5):
    """
    Calculate permutation feature importance
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        feature_names: List of feature names
        model_type: 'sklearn', 'xgboost', or 'rnn'
        n_repeats: Number of permutation repeats
    
    Returns:
        DataFrame with feature importance scores
    """
    if model_type == 'rnn':
        # Custom implementation for neural networks
        baseline_score = gini_coefficient(y_test, model.predict(X_test, verbose=0).flatten())
        
        importance_scores = []
        importance_stds = []
        
        for i in range(X_test.shape[1]):
            scores = []
            for _ in range(n_repeats):
                X_permuted = X_test.copy()
                X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
                permuted_pred = model.predict(X_permuted, verbose=0).flatten()
                permuted_score = gini_coefficient(y_test, permuted_pred)
                importance = baseline_score - permuted_score
                scores.append(importance)
            
            importance_scores.append(np.mean(scores))
            importance_stds.append(np.std(scores))
    
    else:
        # For sklearn-compatible models (Logistic Regression and XGBoost)
        if model_type == 'sklearn':
            # For logistic regression with probability output
            scoring_func = lambda est, X, y: gini_coefficient(y, est.predict_proba(X)[:, 1])
        else:
            # For XGBoost with direct prediction
            scoring_func = lambda est, X, y: gini_coefficient(y, np.clip(est.predict(X), 0, 1))
        
        perm_importance = permutation_importance(
            model, X_test, y_test,
            scoring=scoring_func,
            n_repeats=n_repeats,
            random_state=42
        )
        
        importance_scores = perm_importance.importances_mean
        importance_stds = perm_importance.importances_std
    
    # Create results DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': importance_scores,
        'importance_std': importance_stds
    }).sort_values('importance_mean', ascending=False)
    
    return importance_df

import shap
import numpy as np
import pandas as pd

def shap_feature_importance(model, X_test, feature_names, model_type='sklearn'):
    """
    Calculate SHAP feature importance

    Args:
        model: Trained model
        X_test: Test features (numpy or DataFrame)
        feature_names: List of feature names
        model_type: 'sklearn', 'xgboost', or 'rnn'

    Returns:
        DataFrame with mean absolute SHAP value per feature
    """
    # Choose appropriate SHAP explainer
    if model_type == 'xgboost':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

    elif model_type == 'sklearn':
        explainer = shap.Explainer(model, X_test)  # works for tree and linear models
        shap_values = explainer(X_test).values

    elif model_type == 'rnn':
        # Assumes model is a Keras model and X_test is a numpy array
        explainer = shap.DeepExplainer(model, X_test)
        shap_values = explainer.shap_values(X_test)[0]

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Compute mean absolute SHAP value for each feature
    shap_values = np.abs(shap_values)
    mean_shap = np.mean(shap_values, axis=0)

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': mean_shap
    }).sort_values('importance_mean', ascending=False)

    return importance_df


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
    lr_importance = permutation_feature_importance(lr_model, X_test_scaled, y_test, feature_names, 'sklearn')
    print("Top 5 important features:")
    print(lr_importance.head())
    
    # Run XGBoost
    print("\n--- XGBoost ---")
    xgb_model, xgb_pred, xgb_gini = run_xgboost(X_train, y_train, X_test, y_test)
    print(f"Gini coefficient: {xgb_gini:.4f}")
    
    # Calculate feature importance for XGBoost
    xgb_importance = permutation_feature_importance(xgb_model, X_test, y_test, feature_names, 'xgboost')
    print("Top 5 important features:")
    print(xgb_importance.head())
    
    # Run RNN
    print("\n--- RNN ---")
    rnn_model, rnn_pred, rnn_gini, rnn_history = run_rnn(X_train_scaled, y_train, X_test_scaled, y_test)
    print(f"Gini coefficient: {rnn_gini:.4f}")
    
    # Calculate feature importance for RNN
    rnn_importance = permutation_feature_importance(rnn_model, X_test_scaled, y_test, feature_names, 'rnn')
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
        lr_importance = permutation_feature_importance(lr_model, X_test_scaled, y_test, feature_names, 'sklearn')
        results[dataset_name]['logistic'] = {'gini': lr_gini, 'importance': lr_importance}
        print(f"Logistic Regression Gini: {lr_gini:.4f}")
        
        # Run XGBoost
        print("Running XGBoost...")
        xgb_model, xgb_pred, xgb_gini = run_xgboost(X_train, y_train, X_test, y_test)
        xgb_importance = permutation_feature_importance(xgb_model, X_test, y_test, feature_names, 'xgboost')
        results[dataset_name]['xgboost'] = {'gini': xgb_gini, 'importance': xgb_importance}
        print(f"XGBoost Gini: {xgb_gini:.4f}")
        
        # Run RNN
        print("Running RNN...")
        rnn_model, rnn_pred, rnn_gini, rnn_history = run_rnn(X_train_scaled, y_train, X_test_scaled, y_test)
        rnn_importance = permutation_feature_importance(rnn_model, X_test_scaled, y_test, feature_names, 'rnn')
        results[dataset_name]['rnn'] = {'gini': rnn_gini, 'importance': rnn_importance}
        print(f"RNN Gini: {rnn_gini:.4f}")
    
    return results

if __name__ == "__main__":
    # Run the example
    # results = run_all_datasets()
    pass
