import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam

# GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print('GPU is successfully loaded')
print('-------------------------------------')

# Set up paths (assumes the same directory structure as your provided code)
current_directory  = os.path.dirname(__file__)
dataset_path = os.path.join(current_directory, '..', '..', '..', 'datasets')
model_path = os.path.join(current_directory,'..', 'model', 'CNN_LSTM.h5')

# Load and preprocess data
df = pd.read_csv(os.path.join(dataset_path, 'IIoT_formatted.csv'))

'''
def preprocess_data(df):
    X = df.drop(columns=['label'])
    y = df['label']

    # Converting boolean to int
    for column in X.columns:
        if X[column].dtype == bool:
            X[column] = X[column].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Reshape data to fit the model
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    return X_train, X_test, y_train, y_test, label_encoder.classes_
'''
def preprocess_data(df, n_components=0.95):
    X = df.drop(columns=['label'])
    y = df['label']

    # Converting boolean to int
    for column in X.columns:
        if X[column].dtype == bool:
            X[column] = X[column].astype(int)

    # Check for infinite values
    '''
    Identify Columns with Infinite Values ->Replace Infinite Values -> Handle NaN Values -> Data Scaling
    '''
    inf_columns = X.columns.to_series()[np.isinf(X).any()]
    print("Columns with infinite values:", inf_columns)

    # Replace inf and -inf with NaN across the DataFrame
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Fill NaN values with the mean of each column
    for column in X.select_dtypes(include=[np.number]).columns:
        X[column] = X[column].fillna(X[column].mean())


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    

    # Applying PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    print(f'PCA reduced the dimensionality to: {X_pca.shape[1]} features')

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    # Reshape data to fit the model: PCA reduced dimensions still need to be treated as 'channels' for CNN
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    return X_train, X_test, y_train, y_test, label_encoder.classes_

def build_Simplified_CNN_LSTM_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        LSTM(50, return_sequences=False),
        Dropout(0.5),
        Dense(50, activation='relu'),
        Dense(len(set(y_train)), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

X_train, X_test, y_train, y_test, class_labels  = preprocess_data(df)
model = build_Simplified_CNN_LSTM_model((X_train.shape[1], 1))
model.fit(X_train, y_train, epochs=8, batch_size=128, validation_split=0.1)

# Save the model
model.save(model_path)

# Evaluate the model
model = tf.keras.models.load_model(model_path)
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
report = classification_report(y_test, y_pred)
print(report)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', 
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.savefig('CMcl.png')
plt.show()