import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf

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
1. Load data and preprocessing
'''
# Load the dataset
current_directory  = os.path.dirname(__file__)
dataset_path = os.path.join(current_directory, '..', '..', 'dataset')
model_path = os.path.join(current_directory, 'model', 'AE.h5')
df = pd.read_csv(os.path.join(dataset_path, 'filtered_df8.csv'))

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


'''
Building the Autoencoder with L1 Regularization
Adjust the encoder by adding L1 regularization to encourage sparsity:
'''
input_dim = X_train.shape[1]  # Number of input features
encoding_dim = 25  # This is a hyperparameter to be tuned


# Encoder with L1 regularization
input_layer = Input(shape=(input_dim,))
# Include L1 regularization in the dense layer
encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

'''
Train the Autoencoder
'''
autoencoder.fit(X_train, X_train,
                epochs=1,
                batch_size=128,
                shuffle=True,
                validation_data=(X_test, X_test))

'''
Feature Selection from the Encoder
After training, you can examine the weights of the encoder to identify which features are most significant. 
Features corresponding to higher absolute weights in the encoded representation are considered more important.
'''
# Get encoder weights
encoder_weights = encoder.get_weights()[0]  # Weights of the first layer

# Summing the absolute weights to see the importance of each feature
feature_importance = np.sum(np.abs(encoder_weights), axis=1)
feature_ranking = np.argsort(feature_importance)[::-1]  # Descending order

print("Feature importance ranking:", feature_ranking)


'''
Using the Encoded Features for Classification
As before, use the transformed (encoded) features for classification
'''
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)

classifier = Sequential([
    Dense(64, activation='relu', input_dim=encoding_dim),
    Dense(32, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
classifier.fit(X_train_encoded, y_train, epochs=3, batch_size=32, validation_split=0.1)

'''
Evaluate the classifier
'''
y_pred = classifier.predict(X_test_encoded)
y_pred_classes = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_classes))