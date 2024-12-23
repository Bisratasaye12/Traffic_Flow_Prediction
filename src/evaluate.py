import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os
import seaborn as sns
import matplotlib.pyplot as plt

def load_test_data():
    """
    Load the test dataset.
    """
    test_data = pd.read_csv('data/processed/test.csv')
    X_test = test_data.drop(['TrafficLevel'], axis=1)
    y_test = test_data['TrafficLevel']
    return X_test, y_test

def load_models():
    """
    Load all saved models from the models folder.
    """
    model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
    models = {file.split('.')[0]: joblib.load(f'models/{file}') for file in model_files}
    return models

def save_confusion_matrix(conf_matrix, model_name):
    """
    Save confusion matrix as a heatmap image.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.savefig(f'data/processed/confusion_matrix_{model_name}.png')
    plt.close()

def save_classification_report(report, model_name):
    """
    Save the classification report to a text file.
    """
    with open(f'data/processed/classification_report_{model_name}.txt', 'w') as f:
        f.write(report)

def evaluate_models(models, X_test, y_test):
    """
    Evaluate each model on test data.
    """
    metrics = []
    confusion_matrices = {}
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        metrics.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })
        
        confusion_matrix_result = confusion_matrix(y_test, y_pred)
        confusion_matrices[name] = confusion_matrix_result
        
        # Save confusion matrix heatmap as an image
        save_confusion_matrix(confusion_matrix_result, name)
        
        # Generate and save the classification report to a file
        report = classification_report(y_test, y_pred)
        save_classification_report(report, name)
    
    return metrics, confusion_matrices

def main():
    X_test, y_test = load_test_data()
    models = load_models()
    metrics, confusion_matrices = evaluate_models(models, X_test, y_test)
    
    # Save metrics to CSV
    pd.DataFrame(metrics).to_csv('data/processed/metrics.csv', index=False)
    
    return metrics, confusion_matrices

if __name__ == "__main__":
    metrics, confusion_matrices = main()
    # Save metrics for visualization
