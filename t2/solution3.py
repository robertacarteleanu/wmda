import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Load the Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Optional: convert to DataFrame for exploration
# df = pd.DataFrame(X, columns=wine.feature_names)
# df['target'] = y
# print(df.head())

# 2. Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train a Decision Tree Classifier
#    max_depth=3 to control overfitting a bit
dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_classifier.fit(X_train, y_train)

# 4. Check accuracy on the test set
y_pred_dt = dt_classifier.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

print(f"Decision Tree Accuracy: {accuracy_dt:.2f}")

# (Optional) Visualize the tree structure
plt.figure(figsize=(12, 8))
plot_tree(dt_classifier, feature_names=wine.feature_names, class_names=wine.target_names, filled=True)
plt.title("Decision Tree Visualization")
plt.show()


# (Optional) Feature importances
feature_importances = pd.DataFrame({
    "Feature": wine.feature_names,
    "Importance": dt_classifier.feature_importances_
}).sort_values(by="Importance", ascending=False)

print(feature_importances)