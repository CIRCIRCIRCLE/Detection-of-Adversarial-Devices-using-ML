import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, LSTM, TimeDistributed
from tensorflow.keras.layers import Reshape, Lambda
from tensorflow.keras import regularizers
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import json
import logging

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
    '''
    1. Load and preprocess data
    '''
    # current_directory = os.path.dirname(__file__)
    # dataset_path = os.path.join(current_directory, '..', '..', 'dataset')
    # df = pd.read_csv(os.path.join(dataset_path, 'filtered_df.csv'))
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
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(df.drop('label', axis=1), df['label'], test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, label_encoder

def build_autoencoder(input_dim, encoding_dim):
    '''
    2. Building the Autoencoder with L1 Regularization
    Adjust the encoder by adding L1 regularization to encourage sparsity:
    '''
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l2(10e-6))(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

def train_autoencoder(autoencoder, X_train, X_test, epochs, batch_size):
    '''
    3. Train the autoencoder
    '''
    autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(X_test, X_test))
    return autoencoder

def feature_importance(encoder):
    '''
    4. Feature Selection from the Encoder
    After training, you can examine the weights of the encoder to identify which features are most significant. 
    Features corresponding to higher absolute weights in the encoded representation are considered more important.
    '''
    encoder_weights = encoder.get_weights()[0]
    feature_importance = np.sum(np.abs(encoder_weights), axis=1)
    feature_ranking = np.argsort(feature_importance)[::-1]
    return feature_ranking

def build_and_train_classifier(X_train_encoded, y_train, encoding_dim, label_encoder, epochs, batch_size):
    '''
    5. Using the Encoded Features for Classification
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

def CNN_LSTM_classifier(X_train, y_train, encoding_dim, label_encoder, epochs, batch_size):
    model = Sequential()
    # Define the input shape for the TimeDistributed layer
    input_shape = (None, X_train.shape[2], 1)  # 'None' for variable number of timesteps

    # add CNN layers
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))

    # add LSTM layers
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(16))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(len(label_encoder.classes_), activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.05)

    return model

def LSTM_CNN_classifier(X_train, y_train, encoding_dim, label_encoder, epochs, batch_size):
    # LSTM expects input of shape (samples, time steps, features)
    # reshape in the calling function to (None, 1, encoding_dim)
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(encoding_dim, 1)),  # LSTM layer
        Dropout(0.2),
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.05)

    return model

def evaluate_classifier(classifier, X_test_encoded, y_test):
    '''
    6. evaluate the performance of classifier
    '''
    y_pred = classifier.predict(X_test_encoded)
    y_pred_classes = np.argmax(y_pred, axis=1)
    return classification_report(y_test, y_pred_classes)

def visualize_confusion_matrix(y_true, y_pred, classes):
    """
    This function computes and plots a confusion matrix.
    `y_true` are the actual labels, `y_pred` are the model's predictions, and `classes` are the label encoder classes.
    """
    
    cm = confusion_matrix(y_true, y_pred)
    logging.info("Confusion Matrix:\n%s", cm)
    
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt='g', cmap = 'Blues')  
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('ConfusionMatrix.png')

 

def main():
    setup_logging()
    logging.info("Starting the application.")
 
    # load configuration info
    config = load_config()
    setup_gpu(config['gpu'])    
    classifier_type = config['classifier_type']
    encoding_dim = config['encoding_dim']
    datapath = config['dataset_path']
    epochs = config['ClassifierEpochs']
    AEEpochs = config['AEEpochs']
    batch_size = config['batch_size']
    logging.info("Loaded configuration and initialized GPU settings.")

    X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess_data(datapath)
    logging.info("Data loaded and preprocessed successfully.")

    # Train the autoencoder
    autoencoder, encoder = build_autoencoder(X_train.shape[1], encoding_dim)
    autoencoder.summary()
    train_autoencoder(autoencoder, X_train, X_test, AEEpochs, batch_size)

    # feature selection
    ranking = feature_importance(encoder)
    logging.info(f'Total number of features: {len(ranking)}')
    logging.info("Feature importance ranking: %s", ranking)

    # Classification
    X_train_encoded = encoder.predict(X_train)
    X_test_encoded = encoder.predict(X_test)

    # Classification choices
    if classifier_type == "CNN":
        # Specifically for CNN classifier reshaping
        X_train_encoded = X_train_encoded.reshape((X_train_encoded.shape[0], X_train_encoded.shape[1], 1))
        X_test_encoded = X_test_encoded.reshape((X_test_encoded.shape[0], X_test_encoded.shape[1], 1))
        classifier = CNN_classifier(X_train_encoded, y_train, X_train_encoded.shape[1], label_encoder, epochs, batch_size)
    elif classifier_type == "LSTM":
        # Specific for LSTM classifier reshaping
        X_train_encoded = X_train_encoded.reshape((X_train_encoded.shape[0], 1, X_train_encoded.shape[1]))
        X_test_encoded = X_test_encoded.reshape((X_test_encoded.shape[0], 1, X_test_encoded.shape[1]))
        classifier = LSTM_classifier(X_train_encoded, y_train, encoding_dim, label_encoder, epochs, batch_size)
    elif classifier_type == "CNN_LSTM":
        X_train_encoded = X_train_encoded.reshape((X_train_encoded.shape[0], 1, X_train_encoded.shape[1], 1))
        X_test_encoded = X_test_encoded.reshape((X_test_encoded.shape[0], 1, X_test_encoded.shape[1], 1))
        classifier = CNN_LSTM_classifier(X_train_encoded, y_train, X_train_encoded.shape[1], label_encoder, epochs, batch_size)
    elif classifier_type == "LSTM_CNN":
        # Reshape data for LSTM_CNN
        X_train_encoded = X_train_encoded.reshape((X_train_encoded.shape[0], X_train_encoded.shape[1], 1))
        X_test_encoded = X_test_encoded.reshape((X_test_encoded.shape[0], X_test_encoded.shape[1], 1))
        classifier = LSTM_CNN_classifier(X_train_encoded, y_train, X_train_encoded.shape[1], label_encoder, epochs, batch_size)
    else:
        # Default to a basic dense network classifier
        classifier = build_and_train_classifier(X_train_encoded, y_train, encoding_dim, label_encoder, epochs, batch_size)

    y_pred = classifier.predict(X_test_encoded)
    y_pred_classes = np.argmax(y_pred, axis=1)

    report = classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_)
    logging.info(f"Choosed Classifier:{classifier_type}  Selected Features: {encoding_dim}")
    logging.info("Classification report:\n%s", report)    
    
    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes, average='macro')
    recall = recall_score(y_test, y_pred_classes, average='macro')
    f1 = f1_score(y_test, y_pred_classes, average='macro')
    logging.info(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

    visualize_confusion_matrix(y_test, y_pred_classes, label_encoder.classes_)


if __name__ == "__main__":
    main()