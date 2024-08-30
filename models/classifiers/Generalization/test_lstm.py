import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os

# GPU settings
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

# Path settings
current_directory = os.path.dirname(__file__)
dataset_path = os.path.join(current_directory, '..', '..', '..', 'datasets')
model_path = os.path.join(current_directory, '..', 'model', 'CNN.h5')
pca_path = os.path.join(current_directory, '..', 'model', 'pca_cnn.pkl')

# Load new dataset
new_dataset_path = os.path.join(dataset_path, 'CIC_formatted.csv')
try:
    df_new = pd.read_csv(new_dataset_path)
except FileNotFoundError as e:
    print(f"New dataset not found: {e}")
    raise

# Load PCA model
pca = joblib.load(pca_path)

# Preprocessing function for new dataset
def preprocess_data_LSTM(df, pca):
    X = df.drop(columns=['label'])
    y = df['label']

    # Converting boolean to int
    for column in X.columns:
        if X[column].dtype == bool:
            X[column] = X[column].astype(int)

    # Check for infinite values
    inf_columns = X.columns.to_series()[np.isinf(X).any()]
    print("Columns with infinite values:", inf_columns)

    # Replace inf and -inf with NaN across the DataFrame
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Fill NaN values with the mean of each column
    for column in X.select_dtypes(include=[np.number]).columns:
        X[column] = X[column].fillna(X[column].mean())

    # Data scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Applying PCA
    X_pca = pca.transform(X_scaled)
    print(f'PCA transformed the dimensionality to: {X_pca.shape[1]} features')

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    print('label number', label_encoder.classes_)

    # Reshaping the input for LSTM
    X_pca = X_pca.reshape((X_pca.shape[0], 1, X_pca.shape[1]))

    print(X_pca.shape, y.shape)

    return X_pca, y, label_encoder.classes_

# Preprocess new dataset
X_new, y_new, class_labels = preprocess_data_LSTM(df_new, pca)

# Load trained model
model = load_model(model_path)

# Evaluate model on new dataset
y_pred_prob = model.predict(X_new)
y_pred = np.argmax(y_pred_prob, axis=1)
report = classification_report(y_new, y_pred)
print(report)

accuracy = accuracy_score(y_new, y_pred)
precision = precision_score(y_new, y_pred, average='macro')
recall = recall_score(y_new, y_pred, average='macro')
f1 = f1_score(y_new, y_pred, average='macro')
print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

# Generate confusion matrix
cm = confusion_matrix(y_new, y_pred)

# Plotting the confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', 
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.savefig('CMLSTM_new.png')
plt.show()
