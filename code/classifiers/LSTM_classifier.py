import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
print('GPU is successfully loaded')
print('-------------------------------------')


'''
Use LSTM for classification
'''
#path direction---------------------------------------------------------
current_directory  = os.path.dirname(__file__)
dataset_path = os.path.join(current_directory, '..', '..', 'datasets')
model_path = os.path.join(current_directory, 'model', 'LSTMfull.h5')
df8 = pd.read_csv(os.path.join(dataset_path, 'filtered_df.csv'))

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

    return X_train, X_test, y_train, y_test

def LSTM_model(X_train, X_test, y_train, y_test, epochs=10, batch_size=256):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=32, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.Dropout(0.2), 
        tf.keras.layers.LSTM(units=16),
        tf.keras.layers.Dropout(0.2), 
        tf.keras.layers.Dense(len(set(y_train)), activation='softmax')  
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.05)

    return model


X_train, X_test, y_train, y_test = preprocess_data_LSTM(df8)
model = LSTM_model(X_train, X_test, y_train, y_test)
model.save(model_path)

#Test
model = load_model(model_path)
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
report = classification_report(y_test, y_pred)
print(report)