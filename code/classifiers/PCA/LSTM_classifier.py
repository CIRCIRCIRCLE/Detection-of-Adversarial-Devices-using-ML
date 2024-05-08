import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

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


'''
Use LSTM for classification
'''
#path direction---------------------------------------------------------
current_directory  = os.path.dirname(__file__)
dataset_path = os.path.join(current_directory, '..', '..', 'dataset')
model_path = os.path.join(current_directory, 'model', 'LSTM2.h5')
df8 = pd.read_csv(os.path.join(dataset_path, 'df2_test.csv'))  #8： filtered_df.csv

'''
#Preprocessing----------------------------------------------------------------
def preprocess_data_LSTM(df):
    X = df.drop(columns=['label'])
    y = df['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for column in X.columns:
        if X[column].dtype == bool:
            X[column] = X[column].astype(int)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    print('label number', label_encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)  

    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    return X_train, X_test, y_train, y_test, label_encoder.classes_
'''

def preprocess_data_LSTM(df, n_components=0.95):
    X = df.drop(columns=['label'])
    y = df['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Applying PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    print(f'PCA reduced the dimensionality to: {X_pca.shape[1]} features')

    for column in X.columns:
        if X[column].dtype == bool:
            X[column] = X[column].astype(int)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    print('label number', label_encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)  

    # Reshaping the input for LSTM
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    return X_train, X_test, y_train, y_test, label_encoder.classes_


def LSTM_model(X_train, X_test, y_train, y_test, epochs=8, batch_size=128):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.Dropout(0.5), 
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(len(set(y_train)), activation='softmax')  
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.05)

    return model


X_train, X_test, y_train, y_test, class_labels = preprocess_data_LSTM(df8)
model = LSTM_model(X_train, X_test, y_train, y_test)
model.save(model_path)

#Test
model = load_model(model_path)
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
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
plt.savefig('CMLSTM.png')
plt.show()