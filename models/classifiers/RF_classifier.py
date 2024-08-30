import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

'''
Use Random forest for classifiction

    Default Random Forest: n_estimators=100
    Evaluation Metrics: Accuracy, Precision, Recall, F1-score
    including 3 types of classification:
        2 classes: Benign or Malicious
        8 classes: Benign, DoS, DDoS, Recon, Mirai, Spoofing, Web, BruteForce
        34 classes: subgroups of the above 8 classes
'''

current_directory  = os.path.dirname(__file__)
dataset_path = os.path.join(current_directory, '..', '..', 'datasets')
df2 = pd.read_csv(os.path.join(dataset_path, 'df2.csv'))
df8 = pd.read_csv(os.path.join(dataset_path, 'df8.csv'))
df34 = pd.read_csv(os.path.join(dataset_path, 'df34.csv'))

def random_forest_classifier(df, label_column, test_size=0.2, random_state=42, n_estimators=100):
    X = df.drop(columns=[label_column])
    y = df[label_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print('------------------Start training----------------------')

    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)

    accuracy = rf_classifier.score(X_test, y_test)
    print(f"Random Forest Classifier Accuracy on {len(y.unique())} classes: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))


rf2 = random_forest_classifier(df2, 'label')
joblib.dump(rf2, os.path.join(current_directory, 'model', 'rf2.pkl'))

rf8 = random_forest_classifier(df8, 'label')
joblib.dump(rf2, os.path.join(current_directory, 'model', 'rf8.pkl'))

rf34 = random_forest_classifier(df34, 'label')
joblib.dump(rf2, os.path.join(current_directory, 'model', 'rf34.pkl'))