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
    np.random.seed(random_state)
    y_flipped = deepcopy(y)
    num_instances = len(y)
    num_flips = int(flip_rate * num_instances)
    
    flip_indices = np.random.choice(num_instances, size=num_flips, replace=False)
    
    y_flipped[flip_indices] = 1 - y_flipped[flip_indices]
    
    print(f"Label Flipping: Flipped {num_flips} out of {num_instances} labels.")
    return X, y_flipped

def feature_manipulation(X, y, manipulation_rate=0.05, feature_range=(0,1.5), random_state=42):
    np.random.seed(random_state)
    X_manipulated = deepcopy(X)
    num_instances = X.shape[0]
    num_manipulations = int(manipulation_rate * num_instances)
    
    manipulation_indices = np.random.choice(num_instances, size=num_manipulations, replace=False)
    
    scaling_factors = np.random.uniform(low=feature_range[0], high=feature_range[1], size=num_manipulations)
    
    num_features = X.shape[1]
    for idx in range(num_manipulations):
        feature_idx = np.random.randint(0, num_features)
        X_manipulated[manipulation_indices[idx], feature_idx] *= scaling_factors[idx]
    
    print(f"Feature Manipulation: Manipulated {num_manipulations} out of {num_instances} instances.")
    return X_manipulated, y

def backdoor_attack(X, y, trigger_feature_idx=0, trigger_value=999, target_label=1, injection_rate=0.05, random_state=42):
    np.random.seed(random_state)
    X_backdoored = deepcopy(X)
    y_backdoored = deepcopy(y)
    num_instances = len(y)
    num_injections = int(injection_rate * num_instances)
    
    injection_indices = np.random.choice(num_instances, size=num_injections, replace=False)
    
    X_synthetic = deepcopy(X[injection_indices])
    X_synthetic[:, trigger_feature_idx] = trigger_value
    y_synthetic = np.full(num_injections, target_label)
    
    X_backdoored = np.vstack((X_backdoored, X_synthetic))
    y_backdoored = np.concatenate((y_backdoored, y_synthetic))
    
    print(f"Backdoor Attack: Injected {num_injections} backdoor instances with trigger feature index {trigger_feature_idx} set to {trigger_value} and target label {target_label}.")
    return X_backdoored, y_backdoored

def injection_attack(X, y, injection_rate=0.05, random_state=42):
    np.random.seed(random_state)
    X_injected = deepcopy(X)
    y_injected = deepcopy(y)
    num_instances = len(y)
    num_injections = int(injection_rate * num_instances)
    
    synthetic_indices = np.random.choice(num_instances, size=num_injections, replace=False)
    noise = np.random.normal(loc=0, scale=0.1, size=(num_injections, X.shape[1]))
    X_synthetic = X[synthetic_indices] + noise
    
    y_synthetic = 1 - y[synthetic_indices]
    
    X_injected = np.vstack((X_injected, X_synthetic))
    y_injected = np.concatenate((y_injected, y_synthetic))
    
    print(f"Injection Attack: Injected {num_injections} new instances into the dataset.")
    return X_injected, y_injected

def retrain_models(X_train_new, y_train_new, best_rf_params, best_xgb_params, scale_pos_weight):
    rf_new = RandomForestClassifier(**best_rf_params, random_state=42, class_weight='balanced')
    xgb_new = XGBClassifier(**best_xgb_params, random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
    
    print("Training RandomForest_Poisoned...")
    rf_new.fit(X_train_new, y_train_new)
    print("RandomForest_Poisoned Training Completed.\n")
    
    print("Training XGBoost_Poisoned...")
    xgb_new.fit(X_train_new, y_train_new)
    print("XGBoost_Poisoned Training Completed.\n")
    
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
    
    models_dict = {
        'RandomForest_Poisoned': rf_new,
        'XGBoost_Poisoned': xgb_new,
        'VotingClassifier_Poisoned': voting_clf_new
    }
    
    return models_dict

def evaluate_models(models_dict, X_val, y_val, evaluation_name):
    results_dict = {}
    for model_name, model in models_dict.items():
        print(f"Evaluating {model_name} on Validation Set ({evaluation_name})...")
        
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:,1]
        
        accuracy = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_proba)
        f1 = f1_score(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred)
        report = classification_report(y_val, y_pred, output_dict=True)
        
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
    results_dict = {}
    for model_name, model in models_dict.items():
        print(f"Evaluating {model_name} on Test Set ({evaluation_name})...")
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
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
    results_df = pd.DataFrame(results_dict).T[['Accuracy', 'AUC', 'F1 Score']]
    
    metrics_table_path = tables_dir / f'{evaluation_type.lower()}_data_results.csv'
    
    results_df.to_csv(metrics_table_path, index=True)
    print(f"{evaluation_type} Data Training Results Saved to {metrics_table_path}\n")
    
    for model_name, metrics in results_dict.items():
        report_json_path = classification_reports_dir / f'{model_name}_{evaluation_type}_classification_report.json'
        with open(report_json_path, 'w') as f:
            json.dump(metrics['Classification Report'], f, indent=4)
        print(f"Classification Report for {model_name} saved to {report_json_path}")
        
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
    results_df = pd.DataFrame(results_dict).T[['Accuracy', 'AUC', 'F1 Score']]
    
    metrics_table_path = tables_dir / f'{evaluation_type.lower()}_test_data_results.csv'
    
    results_df.to_csv(metrics_table_path, index=True)
    print(f"{evaluation_type} Test Data Training Results Saved to {metrics_table_path}\n")
    
    for model_name, metrics in results_dict.items():
        report_json_path = classification_reports_dir / f'{model_name}_{evaluation_type}_test_classification_report.json'
        with open(report_json_path, 'w') as f:
            json.dump(metrics['Classification Report'], f, indent=4)
        print(f"Test Classification Report for {model_name} saved to {report_json_path}")
        
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
    results_df = pd.DataFrame(results_dict).T[['Accuracy', 'AUC']]
    
    accuracy_plot_path = plots_dir / f'{evaluation_type.lower()}_accuracy.png'
    auc_plot_path = plots_dir / f'{evaluation_type.lower()}_auc.png'
    
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
    results_df = pd.DataFrame(results_dict).T[['Accuracy', 'AUC']]
    
    accuracy_plot_path = plots_dir / f'{evaluation_type.lower()}_test_accuracy.png'
    auc_plot_path = plots_dir / f'{evaluation_type.lower()}_test_auc.png'
    
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
    for model_name, model in models_dict.items():
        model_filename = f"{model_name}_{evaluation_type}.joblib"
        
        model_path = models_dir / model_filename
        
        joblib.dump(model, model_path)
        
        print(f"Model {model_name} saved to {model_path}")
