from mpi4py import MPI
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# MPI Initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Start measuring total execution time
start_total_time = time.time()

# Load the dataset (only in rank 0)
data_path = '/home/dhpcap/ML_module/project/kagg/Training.csv'
if rank == 0:
    print("Rank 0: Loading data...")
    try:
        training = pd.read_csv(data_path)
        print("Rank 0: Data loaded successfully.")
    except Exception as e:
        print(f"Rank 0: Error loading data: {e}")
        training = None
else:
    training = None

# Broadcast dataset to all processes
print(f"Rank {rank}: Broadcasting training data...")
training = comm.bcast(training, root=0)
print(f"Rank {rank}: Data broadcasted.")

# Function to augment the data
def augment_data(df, num_copies):
    augmented_data = df.copy()
    
    for _ in range(num_copies):
        for column in df.columns[:-1]:  # Exclude 'prognosis'
            if df[column].dtype == 'int64':
                flip_mask = np.random.rand(len(augmented_data)) < 0.1
                augmented_data.loc[flip_mask, column] = 1 - augmented_data.loc[flip_mask, column]
    
    return augmented_data

# Determine data augmentation per process
original_size = len(training)
target_size = 10000
num_copies = target_size // (original_size * size)  # Split across MPI processes

# Each process augments its portion
df_split = np.array_split(training, size)[rank]
print(f"Rank {rank}: Augmenting data...")
augmented_data = augment_data(df_split, num_copies)
print(f"Rank {rank}: Data augmented.")

# Gather augmented data at rank 0
print(f"Rank {rank}: Gathering augmented data...")
augmented_data_all = comm.gather(augmented_data, root=0)

if rank == 0:
    print("Rank 0: Concatenating augmented data...")
    expanded_df = pd.concat(augmented_data_all, ignore_index=True)
    expanded_df.to_csv('/home/dhpcap/ML_module/project/kagg/Expanded_par_Training.csv', index=False)
    print("Rank 0: Augmented data saved to CSV.")
else:
    expanded_df = None

# Broadcast expanded dataset
print(f"Rank {rank}: Broadcasting expanded dataset...")
temp_data = comm.bcast(expanded_df, root=0)
print(f"Rank {rank}: Expanded dataset broadcasted.")

# Label encoding - Fit LabelEncoder in rank 0 and broadcast it
if rank == 0:
    print("Rank 0: Fitting LabelEncoder...")
    label_encoder = LabelEncoder()
    label_encoder.fit(temp_data['prognosis'])  # Fit only once on the entire 'prognosis' column
else:
    label_encoder = None

# Broadcast the fitted LabelEncoder to all processes
label_encoder = comm.bcast(label_encoder, root=0)

# Encode labels
y_encoded = label_encoder.transform(temp_data['prognosis'])  # Use transform after fitting
X = temp_data.drop(['prognosis'], axis=1)

# Split dataset (same for all processes)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced', None]  # Use balanced class weights
}

# Initialize the Random Forest Classifier
rf = RandomForestClassifier(random_state=42)

# Use GridSearchCV for hyperparameter tuning
print(f"Rank {rank}: Starting GridSearchCV...")
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best estimator
best_rf = grid_search.best_estimator_

# Train the best model
start_train_time = time.time()
print(f"Rank {rank}: Training the best model...")
best_rf.fit(X_train, y_train)
train_time = time.time() - start_train_time
print(f"Rank {rank}: Model trained.")

# Evaluate model
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Gather accuracy metrics at rank 0
accuracy_all = comm.gather(accuracy, root=0)
train_time_all = comm.gather(train_time, root=0)

# Prepare prediction data (e.g., df2 should be a test sample from X_test)
df2 = pd.DataFrame(columns=X_test.columns)
df2 = pd.concat([df2, X_test.iloc[[7]]], axis=0, ignore_index=True)  # Use test set's eighth row for prediction
df2 = df2.fillna(0)  # Ensure no missing values in df2

# Make prediction using the trained model
m = best_rf.predict(df2)

# Inverse transform the predicted labels using the fitted label_encoder
dis = label_encoder.inverse_transform(m)

# Gathering results at rank 0
if rank == 0:
    # Calculate average accuracy and training time across all processes
    avg_accuracy = np.mean(accuracy_all)
    avg_train_time = np.mean(train_time_all)
    
    # Print model evaluation results
    print("\n--- Parallel Model Evaluation Results ---\n")
    print(f"Average Accuracy: {avg_accuracy:.4f}\n")
    print(f"Average Training Time: {avg_train_time:.4f} seconds\n")
    print("\nClassification Report:\n\n", class_report)

    # Print prediction for rank 0
    print('You should do test of : ', dis)
    print('\n\n')
else:
    # Print result from other ranks (optional)
    print(f"Rank {rank} - Accuracy: {accuracy:.4f}, Training Time: {train_time:.4f} seconds")
    print('\n\n')
