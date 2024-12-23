import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_data():
    """
    Load the processed train, validation, and test datasets.
    """
    train_data = pd.read_csv('data/processed/train.csv')
    validation_data = pd.read_csv('data/processed/validation.csv')
    test_data = pd.read_csv('data/processed/test.csv')
    
    # Separate features and target label
    X_train = train_data.drop(['TrafficLevel'], axis=1)
    y_train = train_data['TrafficLevel']
    
    X_validation = validation_data.drop(['TrafficLevel'], axis=1)
    y_validation = validation_data['TrafficLevel']
    
    X_test = test_data.drop(['TrafficLevel'], axis=1)
    y_test = test_data['TrafficLevel']
    
    return X_train, y_train, X_validation, y_validation, X_test, y_test

def train_models(X_train, y_train):
    """
    Train multiple classifiers: Logistic Regression, SVM, KNN, Decision Tree, and Random Forest.
    Save each trained model.
    """
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(solver="saga",max_iter=1000),
        'SVM': SVC(kernel="linear"),
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier()
    }
    
    trained_models = {}
    # print(X_train.isnull().sum())  # Check for missing values column-wise
    # print(X_train.shape)          # Verify dataset dimensions

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        trained_models[model_name] = model
        # Save the model
        joblib.dump(model, f'models/{model_name.lower().replace(" ", "_")}.pkl')
    
    return trained_models

def evaluate_models(trained_models, X_test, y_test):
    """
    Evaluate each trained model on the test set and print classification reports.
    """
    for model_name, model in trained_models.items():
        print(f"Evaluating {model_name}...")
        y_pred = model.predict(X_test)
        print(f"Classification Report for {model_name}:\n")
        print(classification_report(y_test, y_pred))

def main():
    # Load the preprocessed data
    X_train, y_train, X_validation, y_validation, X_test, y_test = load_data()
    
    # Train models and save them
    trained_models = train_models(X_train, y_train)
    
    # Evaluate the trained models
    evaluate_models(trained_models, X_test, y_test)

if __name__ == "__main__":
    main()
