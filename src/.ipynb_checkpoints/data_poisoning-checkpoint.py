# src/data_poisoning.py

import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score, confusion_matrix
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

def label_flipping(X, y, flip_rate=0.05, random_state=42):
    """
    Flip the labels of a specified percentage of instances.

    Parameters:
    - X (np.ndarray): Feature matrix.
    - y (np.ndarray): Target vector.
    - flip_rate (float): Proportion of labels to flip (default 5%).
    - random_state (int): Seed for reproducibility.

    Returns:
    - X_flipped (np.ndarray): Feature matrix (unchanged).
    - y_flipped (np.ndarray): Target vector with flipped labels.
    """
    np.random.seed(random_state)
    y_flipped = deepcopy(y)
    num_instances = len(y)
    num_flips = int(flip_rate * num_instances)
    
    # Randomly select indices to flip
    flip_indices = np.random.choice(num_instances, size=num_flips, replace=False)
    
    # Flip labels: assuming binary classification (0 <-> 1)
    y_flipped[flip_indices] = 1 - y_flipped[flip_indices]
    
    print(f"Label Flipping: Flipped {num_flips} out of {num_instances} labels.")
    return X, y_flipped

def feature_manipulation(X, y, manipulation_rate=0.05, feature_range=(0,1.5), random_state=42):
    """
    Manipulate feature values of a specified percentage of instances.

    Parameters:
    - X (np.ndarray): Feature matrix.
    - y (np.ndarray): Target vector.
    - manipulation_rate (float): Proportion of instances to manipulate (default 5%).
    - feature_range (tuple): Range to scale/manipulate features.
    - random_state (int): Seed for reproducibility.

    Returns:
    - X_manipulated (np.ndarray): Feature matrix with manipulated features.
    - y (np.ndarray): Target vector (unchanged).
    """
    np.random.seed(random_state)
    X_manipulated = deepcopy(X)
    num_instances = X.shape[0]
    num_manipulations = int(manipulation_rate * num_instances)
    
    # Randomly select indices to manipulate
    manipulation_indices = np.random.choice(num_instances, size=num_manipulations, replace=False)
    
    # For simplicity, we'll scale selected features by a factor within the specified range
    scaling_factors = np.random.uniform(low=feature_range[0], high=feature_range[1], size=num_manipulations)
    
    # Select random features to manipulate for each selected instance
    num_features = X.shape[1]
    for idx in range(num_manipulations):
        feature_idx = np.random.randint(0, num_features)
        X_manipulated[manipulation_indices[idx], feature_idx] *= scaling_factors[idx]
    
    print(f"Feature Manipulation: Manipulated {num_manipulations} out of {num_instances} instances.")
    return X_manipulated, y

def backdoor_attack(X, y, trigger_feature_idx=0, trigger_value=999, target_label=1, injection_rate=0.05, random_state=42):
    """
    Perform a backdoor attack by inserting instances with a specific trigger and target label.

    Parameters:
    - X (np.ndarray): Original feature matrix.
    - y (np.ndarray): Original target vector.
    - trigger_feature_idx (int): Index of the feature to set as the trigger.
    - trigger_value (int/float): Value to set in the trigger feature(s).
    - target_label (int): Label to assign to backdoor instances.
    - injection_rate (float): Proportion of data to inject (default 5%).
    - random_state (int): Seed for reproducibility.

    Returns:
    - X_backdoored (np.ndarray): Feature matrix with backdoor instances.
    - y_backdoored (np.ndarray): Target vector with backdoor labels.
    """
    np.random.seed(random_state)
    X_backdoored = deepcopy(X)
    y_backdoored = deepcopy(y)
    num_instances = len(y)
    num_injections = int(injection_rate * num_instances)
    
    # Randomly select indices to inject backdoor instances
    injection_indices = np.random.choice(num_instances, size=num_injections, replace=False)
    
    # Create backdoor instances
    X_synthetic = deepcopy(X[injection_indices])
    X_synthetic[:, trigger_feature_idx] = trigger_value  # Set trigger
    y_synthetic = np.full(num_injections, target_label)  # Assign target label
    
    # Append backdoor instances to the dataset
    X_backdoored = np.vstack((X_backdoored, X_synthetic))
    y_backdoored = np.concatenate((y_backdoored, y_synthetic))
    
    print(f"Backdoor Attack: Injected {num_injections} backdoor instances with trigger feature index {trigger_feature_idx} set to {trigger_value} and target label {target_label}.")
    return X_backdoored, y_backdoored

def injection_attack(X, y, injection_rate=0.05, random_state=42):
    """
    Inject new, malicious instances into the dataset.

    Parameters:
    - X (np.ndarray): Original feature matrix.
    - y (np.ndarray): Original target vector.
    - injection_rate (float): Proportion of data to inject (default 5%).
    - random_state (int): Seed for reproducibility.

    Returns:
    - X_injected (np.ndarray): Feature matrix with injected instances.
    - y_injected (np.ndarray): Target vector with injected labels.
    """
    np.random.seed(random_state)
    X_injected = deepcopy(X)
    y_injected = deepcopy(y)
    num_instances = len(y)
    num_injections = int(injection_rate * num_instances)
    
    # Generate synthetic malicious instances
    # For simplicity, we'll generate them by adding small noise to existing instances
    synthetic_indices = np.random.choice(num_instances, size=num_injections, replace=False)
    noise = np.random.normal(loc=0, scale=0.1, size=(num_injections, X.shape[1]))
    X_synthetic = X[synthetic_indices] + noise
    
    # Assign malicious labels (e.g., flip labels)
    y_synthetic = 1 - y[synthetic_indices]
    
    # Append synthetic data to the original dataset
    X_injected = np.vstack((X_injected, X_synthetic))
    y_injected = np.concatenate((y_injected, y_synthetic))
    
    print(f"Injection Attack: Injected {num_injections} new instances into the dataset.")
    return X_injected, y_injected

def retrain_models(X_train_new, y_train_new, best_rf_params, best_xgb_params, scale_pos_weight):
    """
    Retrain models with new training data.

    Parameters:
    - X_train_new (np.ndarray): New training feature matrix.
    - y_train_new (np.ndarray): New training target vector.
    - best_rf_params (dict): Best parameters for RandomForestClassifier.
    - best_xgb_params (dict): Best parameters for XGBClassifier.
    - scale_pos_weight (float): Scale_pos_weight parameter for XGBClassifier.

    Returns:
    - models_dict (dict): Dictionary containing trained models.
    """
    # Initialize models with best parameters
    rf_new = RandomForestClassifier(**best_rf_params, random_state=42, class_weight='balanced')
    xgb_new = XGBClassifier(**best_xgb_params, random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
    
    # Train the models
    print("Training RandomForest_Poisoned...")
    rf_new.fit(X_train_new, y_train_new)
    print("RandomForest_Poisoned Training Completed.\n")
    
    print("Training XGBoost_Poisoned...")
    xgb_new.fit(X_train_new, y_train_new)
    print("XGBoost_Poisoned Training Completed.\n")
    
    # Initialize VotingClassifier with the newly trained models
    voting_clf_new = VotingClassifier(
        estimators=[
            ('rf', rf_new),
            ('xgb', xgb_new)
        ],
        voting='soft'
    )
    
    print("Training VotingClassifier_Poisoned...")
    voting_clf_new.fit(X_train_new, y_train_new)
    print("VotingClassifier_Poisoned Training Completed.\n")
    
    # Compile models into a dictionary
    models_dict = {
        'RandomForest_Poisoned': rf_new,
        'XGBoost_Poisoned': xgb_new,
        'VotingClassifier_Poisoned': voting_clf_new
    }
    
    return models_dict

def evaluate_models(models_dict, X_val, y_val, evaluation_name):
    """
    Evaluate models and store their performance metrics.

    Parameters:
    - models_dict (dict): Dictionary containing models to evaluate.
    - X_val (np.ndarray): Validation feature matrix.
    - y_val (np.ndarray): Validation target vector.
    - evaluation_name (str): Name identifier for the evaluation (e.g., 'LabelFlipping').

    Returns:
    - results_dict (dict): Dictionary containing evaluation metrics for each model.
    """
    results_dict = {}
    for model_name, model in models_dict.items():
        print(f"Evaluating {model_name} on Validation Set ({evaluation_name})...")
        
        # Predictions
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:,1]
        
        # Evaluation Metrics
        accuracy = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_proba)
        f1 = f1_score(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred)
        report = classification_report(y_val, y_pred, output_dict=True)
        
        # Store results
        results_dict[model_name] = {
            'Accuracy': accuracy,
            'AUC': auc,
            'F1 Score': f1,
            'Confusion Matrix': cm,
            'Classification Report': report
        }
        
        print(f"{model_name} Evaluation Completed.\n")
    
    return results_dict

def evaluate_models_test(models_dict, X_test, y_test, evaluation_name):
    """
    Evaluate models on the test set and store their performance metrics.

    Parameters:
    - models_dict (dict): Dictionary containing models to evaluate.
    - X_test (np.ndarray): Test feature matrix.
    - y_test (np.ndarray): Test target vector.
    - evaluation_name (str): Name identifier for the evaluation (e.g., 'LabelFlipping').

    Returns:
    - results_dict (dict): Dictionary containing evaluation metrics for each model.
    """
    results_dict = {}
    for model_name, model in models_dict.items():
        print(f"Evaluating {model_name} on Test Set ({evaluation_name})...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]
        
        # Evaluation Metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Store results
        results_dict[model_name] = {
            'Accuracy': accuracy,
            'AUC': auc,
            'F1 Score': f1,
            'Confusion Matrix': cm,
            'Classification Report': report
        }
        
        print(f"{model_name} Evaluation Completed.\n")
    
    return results_dict

def save_evaluation_metrics(results_dict, evaluation_type, tables_dir, classification_reports_dir, confusion_matrices_dir):
    """
    Save evaluation metrics, classification reports, and confusion matrices.

    Parameters:
    - results_dict (dict): Dictionary containing evaluation metrics.
    - evaluation_type (str): Type of poisoning ('LabelFlipping', 'FeatureManipulation', etc.).
    - tables_dir (Path): Directory to save CSV tables.
    - classification_reports_dir (Path): Directory to save JSON classification reports.
    - confusion_matrices_dir (Path): Directory to save confusion matrix plots.

    Returns:
    - None
    """
    # Create a DataFrame for metrics
    results_df = pd.DataFrame(results_dict).T[['Accuracy', 'AUC', 'F1 Score']]
    
    # Define the path to save the metrics table
    metrics_table_path = tables_dir / f'{evaluation_type.lower()}_data_results.csv'
    
    # Save the metrics table
    results_df.to_csv(metrics_table_path, index=True)
    print(f"{evaluation_type} Data Training Results Saved to {metrics_table_path}\n")
    
    # Save Classification Reports and Confusion Matrices
    for model_name, metrics in results_dict.items():
        # Save classification report
        report_json_path = classification_reports_dir / f'{model_name}_{evaluation_type}_classification_report.json'
        with open(report_json_path, 'w') as f:
            json.dump(metrics['Classification Report'], f, indent=4)
        print(f"Classification Report for {model_name} saved to {report_json_path}")
        
        # Save confusion matrix plot
        cm = metrics['Confusion Matrix']
        cm_df = pd.DataFrame(cm, index=['Bad', 'Good'], columns=['Predicted Bad', 'Predicted Good'])
        
        plt.figure(figsize=(6,4))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name} ({evaluation_type})')
        plt.ylabel('Actual Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()
        
        cm_plot_path = confusion_matrices_dir / f'{model_name}_{evaluation_type}_confusion_matrix.png'
        plt.savefig(cm_plot_path)
        plt.close()
        print(f"Confusion Matrix for {model_name} saved to {cm_plot_path}\n")

def save_test_evaluation_metrics(results_dict, evaluation_type, tables_dir, classification_reports_dir, confusion_matrices_dir):
    """
    Save test evaluation metrics, classification reports, and confusion matrices.

    Parameters:
    - results_dict (dict): Dictionary containing test evaluation metrics.
    - evaluation_type (str): Type of poisoning ('LabelFlipping', 'FeatureManipulation', etc.).
    - tables_dir (Path): Directory to save CSV tables.
    - classification_reports_dir (Path): Directory to save JSON classification reports.
    - confusion_matrices_dir (Path): Directory to save confusion matrix plots.

    Returns:
    - None
    """
    # Create a DataFrame for metrics
    results_df = pd.DataFrame(results_dict).T[['Accuracy', 'AUC', 'F1 Score']]
    
    # Define the path to save the metrics table
    metrics_table_path = tables_dir / f'{evaluation_type.lower()}_test_data_results.csv'
    
    # Save the metrics table
    results_df.to_csv(metrics_table_path, index=True)
    print(f"{evaluation_type} Test Data Training Results Saved to {metrics_table_path}\n")
    
    # Save Classification Reports and Confusion Matrices
    for model_name, metrics in results_dict.items():
        # Save classification report
        report_json_path = classification_reports_dir / f'{model_name}_{evaluation_type}_test_classification_report.json'
        with open(report_json_path, 'w') as f:
            json.dump(metrics['Classification Report'], f, indent=4)
        print(f"Test Classification Report for {model_name} saved to {report_json_path}")
        
        # Save confusion matrix plot
        cm = metrics['Confusion Matrix']
        cm_df = pd.DataFrame(cm, index=['Bad', 'Good'], columns=['Predicted Bad', 'Predicted Good'])
        
        plt.figure(figsize=(6,4))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name} ({evaluation_type} Test)')
        plt.ylabel('Actual Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()
        
        cm_plot_path = confusion_matrices_dir / f'{model_name}_{evaluation_type}_test_confusion_matrix.png'
        plt.savefig(cm_plot_path)
        plt.close()
        print(f"Test Confusion Matrix for {model_name} saved to {cm_plot_path}\n")

def plot_performance(results_dict, evaluation_type, plots_dir):
    """
    Plot and save Accuracy and AUC bar charts for poisoned models.

    Parameters:
    - results_dict (dict): Dictionary containing evaluation metrics.
    - evaluation_type (str): Type of poisoning ('LabelFlipping', 'FeatureManipulation', etc.).
    - plots_dir (Path): Directory to save plots.

    Returns:
    - None
    """
    # Create a DataFrame for metrics
    results_df = pd.DataFrame(results_dict).T[['Accuracy', 'AUC']]
    
    # Define plot paths
    accuracy_plot_path = plots_dir / f'{evaluation_type.lower()}_accuracy.png'
    auc_plot_path = plots_dir / f'{evaluation_type.lower()}_auc.png'
    
    # Plot Accuracy
    plt.figure(figsize=(8,6))
    sns.barplot(x=results_df.index, y='Accuracy', data=results_df, palette='viridis')
    plt.title(f'Model Accuracy on {evaluation_type} Data')
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(accuracy_plot_path)
    plt.close()
    print(f"{evaluation_type} Accuracy Plot saved to {accuracy_plot_path}\n")
    
    # Plot AUC
    plt.figure(figsize=(8,6))
    sns.barplot(x=results_df.index, y='AUC', data=results_df, palette='magma')
    plt.title(f'Model AUC on {evaluation_type} Data')
    plt.ylabel('AUC')
    plt.xlabel('Model')
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(auc_plot_path)
    plt.close()
    print(f"{evaluation_type} AUC Plot saved to {auc_plot_path}\n")

def plot_test_performance(results_dict, evaluation_type, plots_dir):
    """
    Plot and save Accuracy and AUC bar charts for poisoned models on the test set.

    Parameters:
    - results_dict (dict): Dictionary containing evaluation metrics.
    - evaluation_type (str): Type of poisoning ('LabelFlipping', 'FeatureManipulation', etc.).
    - plots_dir (Path): Directory to save plots.

    Returns:
    - None
    """
    # Create a DataFrame for metrics
    results_df = pd.DataFrame(results_dict).T[['Accuracy', 'AUC']]
    
    # Define plot paths
    accuracy_plot_path = plots_dir / f'{evaluation_type.lower()}_test_accuracy.png'
    auc_plot_path = plots_dir / f'{evaluation_type.lower()}_test_auc.png'
    
    # Plot Accuracy
    plt.figure(figsize=(8,6))
    sns.barplot(x=results_df.index, y='Accuracy', data=results_df, palette='viridis')
    plt.title(f'Model Accuracy on {evaluation_type} Test Data')
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(accuracy_plot_path)
    plt.close()
    print(f"{evaluation_type} Test Accuracy Plot saved to {accuracy_plot_path}\n")
    
    # Plot AUC
    plt.figure(figsize=(8,6))
    sns.barplot(x=results_df.index, y='AUC', data=results_df, palette='magma')
    plt.title(f'Model AUC on {evaluation_type} Test Data')
    plt.ylabel('AUC')
    plt.xlabel('Model')
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(auc_plot_path)
    plt.close()
    print(f"{evaluation_type} Test AUC Plot saved to {auc_plot_path}\n")

def save_poisoned_models(models_dict, evaluation_type, models_dir):
    """
    Save poisoned models using Joblib.

    Parameters:
    - models_dict (dict): Dictionary containing poisoned models.
    - evaluation_type (str): Type of poisoning ('LabelFlipping', 'FeatureManipulation', etc.).
    - models_dir (Path): Directory to save models.

    Returns:
    - None
    """
    for model_name, model in models_dict.items():
        # Define the filename for the model
        model_filename = f"{model_name}_{evaluation_type}.joblib"
        
        # Define the full path to save the model
        model_path = models_dir / model_filename
        
        # Save the model using Joblib
        joblib.dump(model, model_path)
        
        print(f"Model {model_name} saved to {model_path}")
