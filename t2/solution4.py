import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Spambase dataset
#    Make sure spambase.csv is in your current working directory or provide the full path.
df = pd.read_csv("spambase.data", header=None)

# The Spambase dataset typically has:
# - 57 numeric columns for features
# - 1 numeric column (the 58th) for the label (1 = spam, 0 = not spam)

# 2. Separate the features (X) and the target label (y)
X = df.iloc[:, :-1]   # All rows, all columns except the last
y = df.iloc[:, -1]    # All rows, only the last column

# 3. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train a Naïve Bayes classifier (MultinomialNB is common for spam detection)
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# 5. Predict on the test set
y_pred = nb_classifier.predict(X_test)

# 6. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Acuratețe: {accuracy:.2f}')
print(f'Precizie: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print('Matricea de confuzie:')
print(conf_matrix)
