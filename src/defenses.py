# src/defenses.py

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc
)
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def apply_isolation_forest(X, y, contamination=0.05, random_state=42):
    isolation_forest = IsolationForest(n_estimators=100, contamination=contamination, random_state=random_state)
    isolation_forest.fit(X)
    preds = isolation_forest.predict(X)
    mask = preds == 1  # 1 for inliers, -1 for outliers
    X_clean = X[mask]
    y_clean = y[mask]
    print(f"Isolation Forest detected {np.sum(~mask)} anomalies out of {len(X)} instances.")
    return X_clean, y_clean

def apply_lof(X, y, contamination=0.05, n_neighbors=20):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    preds = lof.fit_predict(X)
    mask = preds == 1  # 1 for inliers, -1 for outliers
    X_clean = X[mask]
    y_clean = y[mask]
    print(f"Local Outlier Factor detected {np.sum(~mask)} anomalies out of {len(X)} instances.")
    return X_clean, y_clean

def apply_autoencoder(X, y, encoding_dim=14, epochs=50, batch_size=32, contamination=0.05):
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define Autoencoder architecture
    input_dim = X_scaled.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation="relu", activity_regularizer=regularizers.l1(10e-5))(input_layer)
    decoder = Dense(input_dim, activation='sigmoid')(encoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)

    # Compile and train Autoencoder
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.fit(
        X_scaled, X_scaled,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=0.2,
        verbose=0
    )

    # Predict and calculate reconstruction error
    X_pred = autoencoder.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)

    # Determine threshold for anomaly detection
    threshold = np.percentile(mse, 100 - (contamination * 100))
    print(f"Autoencoder reconstruction error threshold: {threshold}")

    # Identify anomalies
    mask = mse <= threshold
    X_clean = X[mask]
    y_clean = y[mask]
    print(f"Autoencoder detected {np.sum(~mask)} anomalies out of {len(X)} instances.")

    return X_clean, y_clean

def retrain_models_with_params(X, y, best_rf_params, best_xgb_params, scale_pos_weight):
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    # Initialize RandomForest with best parameters
    rf = RandomForestClassifier(**best_rf_params, random_state=42, class_weight='balanced')

    # Initialize XGBoost with best parameters and scale_pos_weight
    xgb = XGBClassifier(
        **best_xgb_params,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight
    )

    # Train RandomForest
    rf.fit(X, y)

    # Train XGBoost
    xgb.fit(X, y)

    return {'RandomForest': rf, 'XGBoost': xgb}

def evaluate_trained_models(models, X, y, evaluation_name='AnomalyDetection'):
    results = {}
    for name, model in models.items():
        print(f"Evaluating {name} on {evaluation_name} scenario...")
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:,1]

        # Compute metrics
        accuracy = accuracy_score(y, y_pred)
        auc_score = roc_auc_score(y, y_proba)
        f1 = f1_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True)

        # Store results
        results[name] = {
            'Accuracy': accuracy,
            'AUC': auc_score,
            'F1 Score': f1,
            'Confusion Matrix': cm,
            'Classification Report': report
        }
        print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}, F1 Score: {f1:.4f}\n")
    return results

def save_evaluation_metrics(results, evaluation_type, tables_dir, classification_reports_dir, confusion_matrices_dir):
    # Create a DataFrame for summary metrics
    summary_df = pd.DataFrame.from_dict(results, orient='index')[['Accuracy', 'AUC', 'F1 Score']]
    
    # Save the summary results
    summary_path = tables_dir / f'{evaluation_type}_results.csv'
    summary_df.to_csv(summary_path, index=True)
    print(f"Summary metrics for {evaluation_type} saved to {summary_path}")
    
    for model_name, metrics in results.items():
        # Save classification report
        classification_report_path = classification_reports_dir / f'{model_name}_{evaluation_type}_classification_report.json'
        with open(classification_report_path, 'w') as f:
            json.dump(metrics['Classification Report'], f, indent=4)
        print(f"Classification report for {model_name} saved to {classification_report_path}")
        
        # Save confusion matrix
        confusion_matrix_path = confusion_matrices_dir / f'{model_name}_{evaluation_type}_confusion_matrix.png'
        cm = metrics['Confusion Matrix']
        cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
        
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name} ({evaluation_type})')
        plt.ylabel('Actual Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()
        plt.savefig(confusion_matrix_path)
        plt.close()
        print(f"Confusion matrix for {model_name} saved to {confusion_matrix_path}")


def plot_roc_curves(models, X, y, plot_path, title='ROC Curves'):
    plt.figure(figsize=(8,6))
    for name, model in models.items():
        y_proba = model.predict_proba(X)[:,1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    plt.plot([0,1], [0,1], 'k--')  # Diagonal line
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"ROC Curves saved to {plot_path}")

def plot_precision_recall_curves(models, X, y, plot_path, title='Precision-Recall Curves'):
    plt.figure(figsize=(8,6))
    for name, model in models.items():
        y_proba = model.predict_proba(X)[:,1]
        precision, recall, _ = precision_recall_curve(y, y_proba)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'{name} (PR AUC = {pr_auc:.2f})')

    plt.title(title)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Precision-Recall Curves saved to {plot_path}")

def plot_feature_importance(models, feature_names, plot_dir):
    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(10,6))
            sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], palette='viridis')
            plt.title(f'Feature Importances - {name}')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            plot_path = plot_dir / 'with_defenses' / f'{name}_feature_importance.png'
            plt.savefig(plot_path)
            plt.close()
            print(f"Feature Importance Plot for {name} saved to {plot_path}")
        else:
            print(f"Model {name} does not have feature_importances_ attribute.")

def compile_comparative_results(clean_results, anomaly_results, evaluation_type):
    
    # Create DataFrame for clean results
    clean_df = pd.DataFrame.from_dict(clean_results, orient='index')[['Accuracy', 'AUC', 'F1 Score']]
    clean_df['Scenario'] = 'Clean Data'

    # Create DataFrame for anomaly results
    anomaly_df = pd.DataFrame.from_dict(anomaly_results, orient='index')[['Accuracy', 'AUC', 'F1 Score']]
    anomaly_df['Scenario'] = evaluation_type

    # Concatenate
    comparative_df = pd.concat([clean_df, anomaly_df], ignore_index=True)

    return comparative_df

def save_comparative_results(comparative_df, plot_path, tables_dir):
    
    # Save to CSV
    scenario = comparative_df['Scenario'].iloc[-1].replace(' ', '_')
    comparative_df.to_csv(tables_dir / 'with_defenses' / f'comparative_results_{scenario}.csv', index=False)
    print(f"Comparative results saved to {tables_dir / 'with_defenses' / f'comparative_results_{scenario}.csv'}")

    # Plotting F1 Scores
    plt.figure(figsize=(10,6))
    sns.barplot(x='Scenario', y='F1 Score', hue=comparative_df.index, data=comparative_df)
    plt.title(f'F1 Score Comparison - {scenario}')
    plt.ylabel('F1 Score')
    plt.xlabel('Scenario')
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Comparative F1 Score plot saved to {plot_path}")
