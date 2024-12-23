# Traffic Level Classification Project

This project aims to predict traffic flow levels in Addis Ababa using machine learning classification algorithms. By leveraging a dataset that includes temporal, spatial, and vehicular metrics such as day of the week, time range, GPS coordinates, mileage, average speed, and total travel time, the goal is to classify traffic conditions into categories like Low Traffic, Medium Traffic, and High Traffic. It trains multiple machine learning models, including Logistic Regression, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Decision Tree, and Random Forest. The project also evaluates these models' performance using various metrics and saves the results for further analysis.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [File Structure](#file-structure)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/traffic-level-classification.git
   cd traffic-level-classification
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file should atleast include:
   ```
   pandas
   scikit-learn
   joblib
   ```

## Usage

To run the project:

```bash
python main.py
```

Or you can run each of the above files independently 

```bash
python <filename>
```



### 1. Data Preprocessing
Data preprocessing is done in the `preprocess.py` file, which loads the train, validation, and test datasets.


This will:
- Load the datasets.
- Split them into features (X) and target labels (y).
- Preprocess and standardize the data using `MinMaxScaler`.

### 2. Model Training
The `train.py` file trains several machine learning models using the training data:
- **Logistic Regression**
- **SVM (Support Vector Machine)**
- **KNN (K-Nearest Neighbors)**
- **Decision Tree**
- **Random Forest**

The models are trained and saved to the `models/` directory in `.pkl` format. You can adjust hyperparameters in the `train.py` file to tune the models.


### 3. Model Evaluation
The `evaluate.py` incorporates functions to loads the trained models and evaluates them on the test dataset. The following metrics are calculated for each model:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

The classification report is saved to `data/processed/`, and the metrics are saved to `data/processed/metrics.csv` for visualisation.

### 4. Model Visualisation
The `visualise.py` includes a visualisation function to plot the evaluated metrices for each algorithm

The evaluation graph is saved at `data/processed/metrics_comparison.png`

## Data Preprocessing

The data is preprocessed before model training. This includes:
1. Loading the train, validation, and test datasets from `data/processed/`.
2. Separating the features (`X`), feature engineering (adding relevant columns such as 'Distanc'), One-Hot Encoding for timeRange, Month and DayOfWeek features and target label (`TrafficLevel`).
3. Standardizing the data using `MinMaxScaler`.

## Model Training

Models are trained using the following algorithms:
- **Logistic Regression**: A linear model used for binary classification.
- **SVM (Support Vector Machine)**: A powerful classifier, using a linear kernel for faster performance on large datasets.
- **KNN (K-Nearest Neighbors)**: A non-parametric method used for classification based on nearest neighbors.
- **Decision Tree**: A decision tree classifier to create interpretable decision-making rules.
- **Random Forest**: An ensemble of decision trees for improved accuracy and generalization.

## Model Evaluation

After training, the models are evaluated using the following metrics:
- **Accuracy**: The proportion of correctly predicted instances.
- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall**: The ratio of correctly predicted positive observations to all observations in the actual class.
- **F1 Score**: The weighted average of Precision and Recall.
- **Confusion Matrix**: A matrix used to evaluate the performance of the classification models.

Results are saved in data/processed folder

## File Structure

```
traffic-level-classification/
.
├── README.md
├── data
│   ├── processed
│   │   ├── classification_report_decision_tree.txt
│   │   ├── classification_report_knn.txt
│   │   ├── classification_report_logistic_regression.txt
│   │   ├── classification_report_random_forest.txt
│   │   ├── classification_report_svm.txt
│   │   ├── confusion_matrix_decision_tree.png
│   │   ├── confusion_matrix_knn.png
│   │   ├── confusion_matrix_logistic_regression.png
│   │   ├── confusion_matrix_random_forest.png
│   │   ├── confusion_matrix_svm.png
│   │   ├── metrics.csv
│   │   ├── metrics_comparison.png
│   │   ├── test.csv
│   │   ├── train.csv
│   │   └── validation.csv
│   └── raw
│       └── data.csv
├── main.py
├── models
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── svm.pkl
├── notebooks
│   └── traffic_flow_analysis.ipynb
├── requirements.txt
└── src
    ├── __pycache__
    │   ├── evaluate.cpython-312.pyc
    │   ├── preprocess.cpython-312.pyc
    │   ├── train.cpython-312.pyc
    │   └── visualise.cpython-312.pyc
    ├── evaluate.py
    ├── preprocess.py
    ├── train.py
    └── visualise.py

Files in the data/processed are generated when running the program.
```
