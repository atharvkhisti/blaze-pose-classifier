import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
import os
from blazepose_classifier import ExerciseType

class ModelEvaluator:
    """Evaluate and compare different models for exercise classification"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        
        self.load_data()
    
    def load_data(self):
        """Load and prepare data"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Dataset loaded: {self.df.shape[0]} samples with {self.df.shape[1]} features")
            
            # Check class distribution
            class_counts = self.df['exercise_type'].value_counts()
            print("Class distribution:")
            for ex_type in ExerciseType:
                count = class_counts.get(ex_type.value, 0)
                print(f"  {ex_type.name}: {count} samples")
                
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return
            
        # Split features and target
        X = self.df.drop('exercise_type', axis=1)
        y = self.df['exercise_type']
        
        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Testing set: {self.X_test.shape[0]} samples")
    
    def train_random_forest(self, params=None, cv=5):
        """Train and evaluate a Random Forest classifier"""
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'random_state': 42
            }
        
        print("\n=== Training Random Forest ===")
        rf = RandomForestClassifier(**params)
        
        # Cross-validation
        cv_scores = cross_val_score(rf, self.X_train, self.y_train, cv=cv)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Average CV score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        # Train on full training set
        rf.fit(self.X_train, self.y_train)
        
        # Evaluate on test set
        y_pred = rf.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Save model
        self.models['random_forest'] = rf
        
        return rf, accuracy
    
    def train_gradient_boosting(self, params=None, cv=5):
        """Train and evaluate a Gradient Boosting classifier"""
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 3,
                'learning_rate': 0.1,
                'random_state': 42
            }
        
        print("\n=== Training Gradient Boosting ===")
        gb = GradientBoostingClassifier(**params)
        
        # Cross-validation
        cv_scores = cross_val_score(gb, self.X_train, self.y_train, cv=cv)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Average CV score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        # Train on full training set
        gb.fit(self.X_train, self.y_train)
        
        # Evaluate on test set
        y_pred = gb.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Save model
        self.models['gradient_boosting'] = gb
        
        return gb, accuracy
    
    def hyperparameter_tuning(self, model_type='random_forest'):
        """Perform hyperparameter tuning with GridSearchCV"""
        if model_type == 'random_forest':
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1, 0.2]
            }
        else:
            print(f"Unknown model type: {model_type}")
            return None
        
        print(f"\n=== Hyperparameter Tuning for {model_type} ===")
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Evaluate on test set
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Test accuracy with tuned parameters: {accuracy:.4f}")
        
        # Save model
        self.models[f'{model_type}_tuned'] = best_model
        
        return best_model, accuracy
    
    def generate_classification_report(self, model_name):
        """Generate a classification report for a model"""
        if model_name not in self.models:
            print(f"Model '{model_name}' not found.")
            return
            
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        
        print(f"\n=== Classification Report for {model_name} ===")
        
        report = classification_report(
            self.y_test, y_pred,
            target_names=[ex_type.name for ex_type in ExerciseType],
            output_dict=True
        )
        
        print(classification_report(
            self.y_test, y_pred,
            target_names=[ex_type.name for ex_type in ExerciseType]
        ))
        
        # Create a DataFrame from the classification report
        report_df = pd.DataFrame(report).transpose()
        
        return report_df
    
    def plot_confusion_matrix(self, model_name):
        """Plot confusion matrix for a model"""
        if model_name not in self.models:
            print(f"Model '{model_name}' not found.")
            return
            
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=[ex_type.name for ex_type in ExerciseType],
            yticklabels=[ex_type.name for ex_type in ExerciseType]
        )
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {model_name}')
        
        # Save the figure
        plt.savefig(f'confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self, model_name, top_n=20):
        """Plot feature importance for a model"""
        if model_name not in self.models:
            print(f"Model '{model_name}' not found.")
            return
            
        model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            print(f"Model '{model_name}' does not have feature importances.")
            return
            
        # Get feature names
        feature_names = list(self.df.drop('exercise_type', axis=1).columns)
        
        # Create DataFrame for feature importance
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        })
        
        # Sort by importance
        importances = importances.sort_values('importance', ascending=False).reset_index(drop=True)
        
        # Plot top N features
        top_features = importances.head(top_n)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title(f'Top {top_n} Feature Importance - {model_name}')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(f'feature_importance_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return importances
    
    def save_model(self, model_name, file_path=None):
        """Save a model to a file"""
        if model_name not in self.models:
            print(f"Model '{model_name}' not found.")
            return False
            
        if file_path is None:
            file_path = f"{model_name}_model.pkl"
            
        model = self.models[model_name]
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Model '{model_name}' saved to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def export_results_to_latex(self, model_name):
        """Export model evaluation results to LaTeX format"""
        if model_name not in self.models:
            print(f"Model '{model_name}' not found.")
            return
            
        # Generate classification report
        report_df = self.generate_classification_report(model_name)
        
        # Format the table for LaTeX
        latex_table = report_df.round(3).to_latex()
        
        # Save to file
        with open(f"{model_name}_results.tex", "w") as f:
            f.write(latex_table)
            
        print(f"LaTeX table saved to {model_name}_results.tex")
        
        return report_df

def main():
    data_path = "exercise_features_dataset.csv"
    
    if not os.path.exists(data_path):
        print(f"Dataset not found: {data_path}")
        print("Please run dataset_generator.py first to collect data.")
        return
    
    # Create results directory if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Change working directory to results
    os.chdir(results_dir)
    
    # Evaluate models
    evaluator = ModelEvaluator(f"../{data_path}")
    
    # Train basic models
    rf_model, rf_accuracy = evaluator.train_random_forest()
    gb_model, gb_accuracy = evaluator.train_gradient_boosting()
    
    # Hyperparameter tuning (optional)
    rf_tuned, rf_tuned_accuracy = evaluator.hyperparameter_tuning('random_forest')
    gb_tuned, gb_tuned_accuracy = evaluator.hyperparameter_tuning('gradient_boosting')
    
    # Generate reports and plots
    for model_name in evaluator.models:
        evaluator.generate_classification_report(model_name)
        evaluator.plot_confusion_matrix(model_name)
        evaluator.plot_feature_importance(model_name)
        evaluator.export_results_to_latex(model_name)
        
    # Save the best model
    best_model_name = max(
        evaluator.models.keys(),
        key=lambda name: accuracy_score(
            evaluator.y_test,
            evaluator.models[name].predict(evaluator.X_test)
        )
    )
    
    print(f"\nBest model: {best_model_name}")
    evaluator.save_model(best_model_name, f"../exercise_classifier_model.pkl")

if __name__ == "__main__":
    main()