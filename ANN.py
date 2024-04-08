import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('diabetes.csv')
print(df.head())

X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ann = MLPClassifier(hidden_layer_sizes=(100, 50, 25), max_iter=500, random_state=42)
ann.fit(X_train_scaled, y_train)
ann_predictions = ann.predict(X_test_scaled)

# Generate and print the classification report and confusion matrix
print(confusion_matrix(y_test, ann_predictions))
print(classification_report(y_test, ann_predictions))