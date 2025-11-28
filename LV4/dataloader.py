import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import keras_tuner as kt

BASE_PATH = "LV4/UCI HAR Dataset"  # <-- change this to your actual path
# 1. Load signal data
def load_signals(subset):
    """
    Input: subset (str) - 'train' or 'test' to specify dataset part
    Process: loads 9 different signal types from multiple text files into numpy arrays
    Output: 3D numpy array of shape (samples, 128 timesteps, 9 signals)
    """
    signal_types = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z"
    ]
    signals = []
    for signal in signal_types:
        filename = os.path.join(BASE_PATH, subset, "Inertial Signals", f"{signal}_{subset}.txt")
        data = np.loadtxt(filename)  # Load text file data as array
        signals.append(data)
        
    # signals is currently list of arrays with shape (samples, 128)
    # We transpose to (samples, 128 timesteps, 9 channels) for LSTM input
    # Shape: (n_samples, 128, 9)
    return np.transpose(signals, (1, 2, 0))

def load_labels(subset):
    """
    Input: subset (str) - 'train' or 'test'
    Process: load labels from file and convert 1-based to 0-based indexing
    Output: 1D numpy array of labels (int), shape (samples,)
    """
    y_path = os.path.join(BASE_PATH, subset, f"y_{subset}.txt")
    labels = np.loadtxt(y_path).astype(int)
    return labels - 1  # convert from 1-6 to 0-5 for zero-based indexing

def load_users(subset):
    """
    Input: subset (str) - 'train' or 'test'
    Process: load users from file and convert 1-based to 0-based indexing
    Output: 1D numpy array of labels (int), shape (samples,)
    """
    subject_path = os.path.join(BASE_PATH, subset, f"subject_{subset}.txt")
    subjects = np.loadtxt(subject_path).astype(int)
    return subjects

X_train = load_signals('train')
X_test = load_signals('test')
y_train = load_labels('train')
y_test = load_labels('test')
subjects_train = load_users('train')
subjects_test = load_users('test')

# One-hot encode labels
y_train_cat = to_categorical(y_train, num_classes=6)
y_test_cat = to_categorical(y_test, num_classes=6)

print(f"Train data shape: {X_train.shape}")  
print(f"Test data shape: {X_test.shape}")
print(f"Train data shape: {y_train.shape}")  
print(f"Test data shape: {y_test.shape}")
print(f"User Train data shape: {subjects_train.shape}")
print(f"User Test data shape: {subjects_test.shape}")

X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train_cat, 
    test_size=0.2, 
    random_state=42, 
    stratify=subjects_train
)

print(f"Final training set shape: {X_train_final.shape}")
print(f"Validation set shape: {X_val.shape}")

