import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class PDPredictionFramework:
    def __init__(self, file_paths, target_columns):
        """
        Initialize the framework with data files and target columns
        
        Args:
            file_paths: List of paths to CSV files
            target_columns: List of target column names for each file
        """
        self.file_paths = file_paths
        self.target_columns = target_columns
        self.datasets = {}
        self.results = {}
        self.models = {}
        self.feature_importance = {}
        
    def gini_coefficient(self, y_true, y_pred):
        """Calculate Gini coefficient from predictions"""
        auc = roc_auc_score(y_true, y_pred)
        return 2 * auc - 1
    
    def load_and_preprocess_data(self):
        """Load and preprocess all datasets"""
        print("Loading and preprocessing datasets...")
        
        for i, (file_path, target_col) in enumerate(zip(self.file_paths, self.target_columns)):
            dataset_name = f"dataset_{i+1}"
            print(f"Processing {dataset_name}: {file_path}")
            
            # Load data
            df = pd.read_csv(file_path)
            
            # Basic info
            print(f"  Shape: {df.shape}")
            print(f"  Target column: {target_col}")
            print(f"  Target range: [{df[target_col].min():.3f}, {df[target_col].max():.3f}]")
            
            # Handle missing values
            df = df.dropna()
            
            # Separate features and target
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Encode categorical variables
            categorical_cols = X.select_dtypes(include=['object']).columns
            label_encoders = {}
            
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                label_encoders[col] = le
            
            # Store processed data
            self.datasets[dataset_name] = {
                'X': X,
                'y': y,
                'categorical_cols': categorical_cols,
                'label_encoders': label_encoders,
                'target_col': target_col
            }
            
            print(f"  Features: {X.shape[1]}")
            print(f"  Categorical features: {len(categorical_cols)}")
            print()
    
    def prepare_model_data(self, X, y, test_size=0.2, random_state=42):
        """Split and scale data for modeling"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler
    
    def train_logistic_regression(self, X_train_scaled, y_train, X_test_scaled, y_test):
        """Train Logistic Regression model"""
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Performance
        gini = self.gini_coefficient(y_test, y_pred_proba)
        
        return model, y_pred_proba, gini
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='rmse'
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        # Clip predictions to [0, 1] range
        y_pred = np.clip(y_pred, 0, 1)
        
        # Performance
        gini = self.gini_coefficient(y_test, y_pred)
        
        return model, y_pred, gini
    
    def train_rnn(self, X_train_scaled, y_train, X_test_scaled, y_test):
        """Train RNN model"""
        # Build model
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
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Predictions
        y_pred = model.predict(X_test_scaled).flatten()
        
        # Performance
        gini = self.gini_coefficient(y_test, y_pred)
        
        return model, y_pred, gini, history
    
    def calculate_permutation_importance(self, model, X_test, y_test, model_type, feature_names):
        """Calculate permutation feature importance"""
        if model_type == 'rnn':
            # For neural networks, we need a custom scoring function
            def neural_net_scorer(model, X, y):
                predictions = model.predict(X).flatten()
                return self.gini_coefficient(y, predictions)
            
            # Manual permutation importance for neural networks
            baseline_score = neural_net_scorer(model, X_test, y_test)
            importance_scores = []
            
            for i in range(X_test.shape[1]):
                X_permuted = X_test.copy()
                X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
                permuted_score = neural_net_scorer(model, X_permuted, y_test)
                importance = baseline_score - permuted_score
                importance_scores.append(importance)
            
            perm_importance = {
                'importances_mean': np.array(importance_scores),
                'importances_std': np.zeros_like(importance_scores)  # Simplified
            }
        else:
            # For sklearn-compatible models
            perm_importance = permutation_importance(
                model, X_test, y_test,
                scoring=lambda estimator, X, y: self.gini_coefficient(y, estimator.predict_proba(X)[:, 1] if hasattr(estimator, 'predict_proba') else estimator.predict(X)),
                n_repeats=5,
                random_state=42
            )
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': perm_importance['importances_mean'],
            'importance_std': perm_importance['importances_std']
        }).sort_values('importance_mean', ascending=False)
        
        return importance_df
    
    def run_experiment(self):
        """Run the complete experiment"""
        print("Starting experiment...")
        
        for dataset_name, data in self.datasets.items():
            print(f"\n{'='*50}")
            print(f"Processing {dataset_name}")
            print('='*50)
            
            X, y = data['X'], data['y']
            
            # Prepare data
            X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = self.prepare_model_data(X, y)
            
            # Initialize results for this dataset
            self.results[dataset_name] = {}
            self.models[dataset_name] = {}
            self.feature_importance[dataset_name] = {}
            
            # Train Logistic Regression
            print("\nTraining Logistic Regression...")
            lr_model, lr_pred, lr_gini = self.train_logistic_regression(
                X_train_scaled, y_train, X_test_scaled, y_test
            )
            
            self.results[dataset_name]['Logistic Regression'] = {
                'gini': lr_gini,
                'complexity': 'Low'
            }
            self.models[dataset_name]['Logistic Regression'] = lr_model
            
            # Calculate permutation importance for LR
            lr_importance = self.calculate_permutation_importance(
                lr_model, X_test_scaled, y_test, 'lr', X.columns
            )
            self.feature_importance[dataset_name]['Logistic Regression'] = lr_importance
            
            print(f"Logistic Regression Gini: {lr_gini:.4f}")
            
            # Train XGBoost
            print("\nTraining XGBoost...")
            xgb_model, xgb_pred, xgb_gini = self.train_xgboost(
                X_train, y_train, X_test, y_test
            )
            
            self.results[dataset_name]['XGBoost'] = {
                'gini': xgb_gini,
                'complexity': 'Medium'
            }
            self.models[dataset_name]['XGBoost'] = xgb_model
            
            # Calculate permutation importance for XGBoost
            xgb_importance = self.calculate_permutation_importance(
                xgb_model, X_test.values, y_test, 'xgb', X.columns
            )
            self.feature_importance[dataset_name]['XGBoost'] = xgb_importance
            
            print(f"XGBoost Gini: {xgb_gini:.4f}")
            
            # Train RNN
            print("\nTraining RNN...")
            rnn_model, rnn_pred, rnn_gini, rnn_history = self.train_rnn(
                X_train_scaled, y_train, X_test_scaled, y_test
            )
            
            self.results[dataset_name]['RNN'] = {
                'gini': rnn_gini,
                'complexity': 'High'
            }
            self.models[dataset_name]['RNN'] = rnn_model
            
            # Calculate permutation importance for RNN
            rnn_importance = self.calculate_permutation_importance(
                rnn_model, X_test_scaled, y_test, 'rnn', X.columns
            )
            self.feature_importance[dataset_name]['RNN'] = rnn_importance
            
            print(f"RNN Gini: {rnn_gini:.4f}")
            
            print(f"\n{dataset_name} Results Summary:")
            print(f"Logistic Regression: {lr_gini:.4f}")
            print(f"XGBoost: {xgb_gini:.4f}")
            print(f"RNN: {rnn_gini:.4f}")
    
    def create_results_summary(self):
        """Create a comprehensive results summary"""
        print("\n" + "="*70)
        print("EXPERIMENT RESULTS SUMMARY")
        print("="*70)
        
        # Create summary dataframe
        summary_data = []
        for dataset_name, models in self.results.items():
            for model_name, metrics in models.items():
                summary_data.append({
                    'Dataset': dataset_name,
                    'Algorithm': model_name,
                    'Gini_Coefficient': metrics['gini'],
                    'Complexity': metrics['complexity']
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Overall performance ranking
        print("\nOverall Performance Ranking (by Gini Coefficient):")
        avg_performance = summary_df.groupby('Algorithm')['Gini_Coefficient'].agg(['mean', 'std']).round(4)
        avg_performance = avg_performance.sort_values('mean', ascending=False)
        print(avg_performance)
        
        # Best model per dataset
        print("\nBest Model per Dataset:")
        for dataset_name in summary_df['Dataset'].unique():
            dataset_results = summary_df[summary_df['Dataset'] == dataset_name]
            best_model = dataset_results.loc[dataset_results['Gini_Coefficient'].idxmax()]
            print(f"{dataset_name}: {best_model['Algorithm']} (Gini: {best_model['Gini_Coefficient']:.4f})")
        
        return summary_df
    
    def plot_results(self):
        """Create visualization of results"""
        summary_df = self.create_results_summary()
        
        # Performance comparison plot
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Gini coefficients by algorithm and dataset
        plt.subplot(2, 2, 1)
        pivot_data = summary_df.pivot(index='Dataset', columns='Algorithm', values='Gini_Coefficient')
        sns.heatmap(pivot_data, annot=True, cmap='YlOrRd', fmt='.4f')
        plt.title('Gini Coefficient by Algorithm and Dataset')
        plt.ylabel('Dataset')
        
        # Subplot 2: Box plot of performance by algorithm
        plt.subplot(2, 2, 2)
        sns.boxplot(data=summary_df, x='Algorithm', y='Gini_Coefficient')
        plt.title('Performance Distribution by Algorithm')
        plt.xticks(rotation=45)
        
        # Subplot 3: Performance vs Complexity
        plt.subplot(2, 2, 3)
        complexity_order = ['Low', 'Medium', 'High']
        complexity_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
        summary_df['Complexity_Numeric'] = summary_df['Complexity'].map(complexity_mapping)
        
        for algorithm in summary_df['Algorithm'].unique():
            alg_data = summary_df[summary_df['Algorithm'] == algorithm]
            plt.scatter(alg_data['Complexity_Numeric'], alg_data['Gini_Coefficient'], 
                       label=algorithm, s=100, alpha=0.7)
        
        plt.xlabel('Model Complexity')
        plt.ylabel('Gini Coefficient')
        plt.title('Performance vs Complexity Trade-off')
        plt.xticks([1, 2, 3], complexity_order)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Average performance with error bars
        plt.subplot(2, 2, 4)
        avg_perf = summary_df.groupby('Algorithm')['Gini_Coefficient'].agg(['mean', 'std'])
        algorithms = avg_perf.index
        means = avg_perf['mean']
        stds = avg_perf['std']
        
        plt.bar(algorithms, means, yerr=stds, capsize=5, alpha=0.7)
        plt.ylabel('Average Gini Coefficient')
        plt.title('Average Performance with Standard Deviation')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def display_feature_importance(self, top_n=10):
        """Display top feature importance for each model and dataset"""
        print("\n" + "="*70)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*70)
        
        for dataset_name, models in self.feature_importance.items():
            print(f"\n{dataset_name}:")
            print("-" * 50)
            
            for model_name, importance_df in models.items():
                print(f"\n{model_name} - Top {top_n} Features:")
                top_features = importance_df.head(top_n)
                for idx, row in top_features.iterrows():
                    print(f"  {row['feature']}: {row['importance_mean']:.4f} (±{row['importance_std']:.4f})")

# Usage Example
if __name__ == "__main__":
    # Define your file paths and target columns
    file_paths = [
        'dataset1.csv',
        'dataset2.csv', 
        'dataset3.csv',
        'dataset4.csv',
        'dataset5.csv'
    ]
    
    target_columns = [
        'target1',  # Replace with your actual target column names
        'target2',
        'target3', 
        'target4',
        'target5'
    ]
    
    # Initialize and run the framework
    framework = PDPredictionFramework(file_paths, target_columns)
    
    # Load and preprocess data
    framework.load_and_preprocess_data()
    
    # Run the complete experiment
    framework.run_experiment()
    
    # Create results summary
    summary_df = framework.create_results_summary()
    
    # Plot results
    framework.plot_results()
    
    # Display feature importance
    framework.display_feature_importance(top_n=10)
    
    # Save results
    summary_df.to_csv('experiment_results_summary.csv', index=False)
    print("\nResults saved to 'experiment_results_summary.csv'")
