# src/model_training.py

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, classification_report

def initialize_models(random_state=42):
    """
    Initialize machine learning models with predefined hyperparameters.

    Parameters:
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: Dictionary of initialized models.
    """
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight='balanced'
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=random_state
        )
    }
    return models

def train_and_evaluate(models, X_train, y_train, X_val, y_val):
    """
    Train and evaluate machine learning models.

    Parameters:
        models (dict): Dictionary of initialized models.
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training target vector.
        X_val (np.ndarray): Validation feature matrix.
        y_val (np.ndarray): Validation target vector.

    Returns:
        dict: Dictionary containing evaluation metrics for each model.
    """
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        print(f"{name} Training Completed.")
        
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
        results[name] = {
            'Accuracy': accuracy,
            'AUC': auc,
            'F1 Score': f1,
            'Confusion Matrix': cm,
            'Classification Report': report
        }
        
        print(f"{name} Evaluation Completed.\n")
    
    return results
