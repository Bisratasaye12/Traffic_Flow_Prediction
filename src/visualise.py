import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import joblib

def load_data():
    """
    Load the processed test dataset for visualization.
    """
    test_data = pd.read_csv('data/processed/test.csv')
    X_test = test_data.drop(['TrafficLevel'], axis=1)
    y_test = test_data['TrafficLevel']
    
    return X_test, y_test

def plot_metrics():
    """
    Plot a bar chart for metrics comparison.
    """
    metrics = pd.read_csv('data/processed/metrics.csv')
    metrics.set_index('Model', inplace=True)
    
    metrics.plot(kind='bar', figsize=(10, 6))
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('data/processed/metrics_comparison.png')
    plt.show()


def main():
   plot_metrics()
       

if __name__ == "__main__":
    main()
