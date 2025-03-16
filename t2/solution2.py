import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# (Optional) Convert to a Pandas DataFrame for easier viewing
# df = pd.DataFrame(X, columns=wine.feature_names)
# df['target'] = y
# print(df.head())  # Uncomment to inspect

# 2. Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train a Naïve Bayes classifier (from Exercise 1)
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)

# 4. Train a Logistic Regression model
log_reg = LogisticRegression(max_iter=5000)
log_reg.fit(X_train, y_train)
y_pred_logreg = log_reg.predict(X_test)

# 5. Compare metrics: accuracy, precision, and recall for each model
# Note: Because we have three classes in the Wine dataset, we set average='macro' (or 'weighted') for multi-class
accuracy_nb = accuracy_score(y_test, y_pred_nb)
precision_nb = precision_score(y_test, y_pred_nb, average='macro')
recall_nb = recall_score(y_test, y_pred_nb, average='macro')

accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
precision_logreg = precision_score(y_test, y_pred_logreg, average='macro')
recall_logreg = recall_score(y_test, y_pred_logreg, average='macro')


# 6. Print results
comparison_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall"],
    "Naïve Bayes": [accuracy_nb, precision_nb, recall_nb],
    "Logistic Regression": [accuracy_logreg, precision_logreg, recall_logreg]
})

print(comparison_df)
# Optional: If you’d like to see a confusion matrix for each model
# from sklearn.metrics import confusion_matrix
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)
conf_matrix_logreg = confusion_matrix(y_test, y_pred_logreg)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(conf_matrix_nb, annot=True, cmap="Blues", xticklabels=wine.target_names, yticklabels=wine.target_names, ax=axes[0])
axes[0].set_title("Naïve Bayes - Confusion Matrix")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

sns.heatmap(conf_matrix_logreg, annot=True, cmap="Blues", xticklabels=wine.target_names, yticklabels=wine.target_names, ax=axes[1])
axes[1].set_title("Logistic Regression - Confusion Matrix")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

plt.show()
