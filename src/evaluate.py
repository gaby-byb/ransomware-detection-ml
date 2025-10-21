
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    matthews_corrcoef,
    precision_recall_curve,
    auc
)
import numpy as np
 
def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate any classifier with Accuracy, MCC, PR-AUC, and Classification Report.
    Automatically handles models without predict_proba().
    """
    print(f"\n--- {model_name.upper()} RESULTS ---")
    
    # Predictions
    y_pred = model.predict(X_test)

    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Try to calculate PR-AUC (only if model has predict_proba or decision_function)
    pr_auc = None
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
        else:
            y_proba = None

        if y_proba is not None:
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            pr_auc = auc(recall, precision)
    except Exception as e:
        print(f"Could not compute PR-AUC: {e}")

    # Print results
    print(f"{'Accuracy:':<20} {accuracy:.3f}")
    print(f"{'MCC:':<20} {mcc:.3f}")
    print(f"{'PR-AUC:':<20} {pr_auc:.3f}" if pr_auc is not None else "PR-AUC: N/A")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("="*60) 