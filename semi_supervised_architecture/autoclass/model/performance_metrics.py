# External
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

def calculate_metrics(y_true, y_pred):
    """ 
        Calculate and print performance metrics 
        Args
        ----------
            Binary classification labels
        Returns
        -------
            Performance metrics
        """
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # F1 Score
    f1 = f1_score(y_true, y_pred)
    print(f"F1 Score: {f1:.4f}")

    return f1, accuracy

def calculate_cm (y_true, y_pred):

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)    

    # Extracting TN, FP, FN, TP from confusion matrix
    TN, FP, FN, TP = cm.ravel()

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Precision
    precision = precision_score(y_true, y_pred)
    print(f"Precision: {precision:.4f}")

    # Recall
    recall = recall_score(y_true, y_pred)
    print(f"Recall: {recall:.4f}")

    # Specificity
    specificity = TN / (TN + FP)
    print(f"Specificity: {specificity:.4f}")

    # F1 Score
    f1 = f1_score(y_true, y_pred)
    print(f"F1 Score: {f1:.4f}")

    # AUROC
    AUROC = roc_auc_score(y_true, y_pred)
    print(f"AUROC: {AUROC:.4f}")
 
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    
