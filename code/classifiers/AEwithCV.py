import numpy as np
import pandas as pd
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, LSTM
from tensorflow.keras import regularizers
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import tensorflow as tf
import json
import logging
import os

# Assuming other functions (load_config, setup_logging, setup_gpu) are defined as before
def load_config():
    with open('config.json', 'r') as file:
        config = json.load(file)
    return config

def setup_logging():
    # Set up logging to file
    logging.basicConfig(
        filename='application.log',  # Log file path
        level=logging.INFO,          # Logging level
        format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
        filemode='w'  # 'w' for overwriting the log file on each run, 'a' for appending
    )
    # Also log to stdout (console)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def setup_gpu(gpu_num="0"):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to only use the specified GPU
            tf.config.experimental.set_visible_devices(gpus[int(gpu_num)], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[int(gpu_num)], True)
            logging.info(f'GPU is successfully loaded on GPU: {gpu_num}')
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logging.error(f"Failed to set up GPU due to: {e}")
    else:
        # No GPU available, use the CPU
        logging.info("No GPU found, using the CPU instead.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # This disables all GPUs

def load_and_preprocess_data(datapath):
    df = pd.read_csv(datapath)
    
    # Convert booleans to integers
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Encode the labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])

    return df, label_encoder

def build_and_train_classifier(X_train_encoded, y_train, encoding_dim, label_encoder, epochs, batch_size):
    '''
    Using the Encoded Features for Classification
    As before, use the transformed (encoded) features for classification
    '''
    classifier = Sequential([
        Dense(64, activation='relu', input_dim=encoding_dim),
        Dense(32, activation='relu'),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    classifier.fit(X_train_encoded, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)
    return classifier

def CNN_classifier(X_train_encoded, y_train, encoding_dim, label_encoder, epochs, batch_size):
    classifier = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(encoding_dim, 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    classifier.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    classifier.summary()
    classifier.fit(X_train_encoded, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.05)
    return classifier

def LSTM_classifier(X_train, y_train, encoding_dim, label_encoder, epochs, batch_size):
    classifier = tf.keras.Sequential([
        LSTM(units=32, return_sequences=True, input_shape=(1, encoding_dim)),
        Dropout(0.2), 
        LSTM(units=16),
        Dropout(0.2), 
        Dense(len(label_encoder.classes_), activation='softmax')
    ])

    classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    classifier.summary()
    classifier.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.05)

    return classifier

def build_autoencoder(input_dim, encoding_dim):
    '''
    2. Building the Autoencoder with L1 Regularization
    Adjust the encoder by adding L1 regularization to encourage sparsity:
    '''
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

def cross_validate_model(data, label_encoder, config):
    # Prepare the data
    X = data.drop('label', axis=1).values
    y = data['label'].values
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_no = 1
    scores = []
    reports = []

    for train, test in kfold.split(X, y):
        # Encoder part
        input_dim = X.shape[1]
        encoding_dim = config['encoding_dim']
        autoencoder, encoder = build_autoencoder(input_dim, encoding_dim)
        autoencoder.fit(X[train], X[train], epochs=config['AEEpochs'], batch_size=config['batch_size'], shuffle=True, validation_split=0.1)

        X_train_encoded = encoder.predict(X[train])
        X_test_encoded = encoder.predict(X[test])

        # Classifier part
        if config['classifier_type'] == "CNN":
            X_train_encoded = X_train_encoded.reshape((-1, encoding_dim, 1))
            X_test_encoded = X_test_encoded.reshape((-1, encoding_dim, 1))
            classifier = CNN_classifier(X_train_encoded, y[train], encoding_dim, label_encoder, config['ClassifierEpochs'], config['batch_size'])
        elif config['classifier_type'] == "LSTM":
            X_train_encoded = X_train_encoded.reshape((-1, 1, encoding_dim))
            X_test_encoded = X_test_encoded.reshape((-1, 1, encoding_dim))
            classifier = LSTM_classifier(X_train_encoded, y[train], encoding_dim, label_encoder, config['ClassifierEpochs'], config['batch_size'])
        else:
            classifier = build_and_train_classifier(X_train_encoded, y[train], encoding_dim, label_encoder, config['ClassifierEpochs'], config['batch_size'])

        # Evaluate and collect results
        y_pred = classifier.predict(X_test_encoded)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Collecting accuracy, precision, recall, and F1-score
        accuracy = accuracy_score(y[test], y_pred_classes)
        precision = precision_score(y[test], y_pred_classes, average='macro')
        recall = recall_score(y[test], y_pred_classes, average='macro')
        f1 = f1_score(y[test], y_pred_classes, average='macro')
        report = classification_report(y[test], y_pred_classes)

        scores.append({
            "fold": fold_no,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

        reports.append(report)
        logging.info(f'Fold {fold_no} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')
        fold_no += 1

    return scores, reports

def main():
    setup_logging()
    logging.info("Starting the application.")

    config = load_config()
    setup_gpu(config['gpu'])  
    logging.info("Configurations are set")
    
    data, label_encoder = load_and_preprocess_data(config['dataset_path'])
    logging.info("Data loaded and preprocessed successfully.")

    results, reports = cross_validate_model(data, label_encoder, config)
    for result in results:
        logging.info(f"Fold {result['fold']} Results: Accuracy={result['accuracy']:.2f}, Precision={result['precision']:.2f}, Recall={result['recall']:.2f}, F1 Score={result['f1_score']:.2f}")

    for idx, report in enumerate(reports, 1):
        logging.info(f"Classification Report for Fold {idx}:\n{report}")


if __name__ == "__main__":
    main()
