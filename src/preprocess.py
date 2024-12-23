import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import MinMaxScaler
from geopy.distance import geodesic
import os

# Constants
INPUT_FILE = 'data/raw/data.csv'
OUTPUT_DIR = 'data/processed'
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(file_path):
    """Load dataset from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess the dataset.

    Steps:
    1. Handle missing values.
    2. Feature engineering (distance and traffic level).
    3. Encode categorical variables.
    4. Normalize numerical features.
    5. Separate features and target label.
    """
    # Handle missing values by dropping rows with NaN values
    data = data.dropna()

    # Feature Engineering: Calculate Distance
    def calculate_distance(row):
        start = (row['Initial latitude'], row['Initial longitude'])
        end = (row['Final latitude'], row['Final longitude'])
        return geodesic(start, end).kilometers

    data.loc[:, 'Distance'] = data.apply(calculate_distance, axis=1)

    # Add Traffic Level (Low, Medium, High) as target label
    def traffic_label(avg_speed):
        if avg_speed > 20:
            return 'Low Traffic'
        elif avg_speed >= 10:
            return 'Medium Traffic'
        else:
            return 'High Traffic'

    data.loc[:, 'TrafficLevel'] = data['Avg_Speed'].apply(traffic_label)

    # Encode Categorical Features
    categorical_columns = ['DayofWeek', 'TimeRange', 'Month']
    encoder = OneHotEncoder(sparse_output=False)
    encoded_cats = pd.DataFrame(encoder.fit_transform(data[categorical_columns]),
                                columns=encoder.get_feature_names_out(categorical_columns))

    # Drop original categorical columns and concatenate encoded ones
    data = pd.concat([data.drop(categorical_columns, axis=1), encoded_cats], axis=1)

    # Normalize Numerical Features
    # scaler = StandardScaler()
    # numerical_columns = ['Mileage', 'total_time', 'Avg_Speed', 'Beginning Time', 'Distance']
    # data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    scaler = MinMaxScaler()
    numerical_columns = ['Mileage', 'total_time', 'Avg_Speed', 'Beginning Time', 'Distance']
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    # Separate Features and Label
    X = data.drop(['TrafficLevel'], axis=1)
    y = data['TrafficLevel']

    return X, y

def split_data(X, y):
    """Split data into training, validation, and test sets."""
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1 - TRAIN_RATIO), random_state=42)
    X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=TEST_RATIO / (VALIDATION_RATIO + TEST_RATIO), random_state=42)

    # Combine features and labels for saving
    train_data = pd.concat([X_train, y_train], axis=1)
    validation_data = pd.concat([X_validation, y_validation], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    return train_data, validation_data, test_data

def save_data(train_data, validation_data, test_data):
    """Save split datasets to files."""
    train_data.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
    validation_data.to_csv(os.path.join(OUTPUT_DIR, 'validation.csv'), index=False)
    test_data.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)

def main():
    """Main function for data preprocessing."""
    print("Loading data...")
    data = load_data(INPUT_FILE)
    print("Data loaded successfully!")

    print("Preprocessing data...")
    X, y = preprocess_data(data)
    print("Preprocessing complete!")

    print("Splitting data...")
    train_data, validation_data, test_data = split_data(X, y)
    print("Data split into training, validation, and test sets.")

    print("Saving processed data...")
    save_data(train_data, validation_data, test_data)
    print(f"Processed datasets saved in {OUTPUT_DIR}.")

if __name__ == "__main__":
    main()
