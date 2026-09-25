# IRIS FLOWER CLASSIFICATION PROJECT

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import joblib
      

# 2. LOAD DATASET

iris = load_iris()

X = iris.data
y = iris.target

print("Dataset loaded successfully!")


# 3. CREATE DATAFRAME FOR EDA

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = iris.target_names[iris.target]

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATASET INFO =====")
print(df.info())

print("\n===== STATISTICS =====")
print(df.describe())


# 4. BASIC EDA VISUALIZATION

sns.pairplot(df, hue="species")
plt.suptitle("Iris Dataset Feature Relationships", y=1.02)
plt.show()


# 5. TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# 6. LOGISTIC REGRESSION MODEL

logistic_model = LogisticRegression(
    max_iter=200,
    random_state=42
)

logistic_model.fit(X_train, y_train)

y_pred_logistic = logistic_model.predict(X_test)


# 7. LOGISTIC REGRESSION EVALUATION

logistic_accuracy = accuracy_score(
    y_test,
    y_pred_logistic
)

print("\n================================")
print("LOGISTIC REGRESSION")

print(
    f"Accuracy: {logistic_accuracy * 100:.2f}%"
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred_logistic
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_logistic,
        target_names=iris.target_names
    )
)


# 8. DECISION TREE MODEL

tree_model = DecisionTreeClassifier(
    random_state=42
)

tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)


# 9. DECISION TREE EVALUATION

tree_accuracy = accuracy_score(
    y_test,
    y_pred_tree
)

print("\n================================")
print("DECISION TREE")

print(
    f"Accuracy: {tree_accuracy * 100:.2f}%"
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred_tree
    )
)


#MODEL COMPARISON
print("\n================================")
print("MODEL COMPARISON")

print(
    f"Logistic Regression: "
    f"{logistic_accuracy * 100:.2f}%"
)

print(
    f"Decision Tree: "
    f"{tree_accuracy * 100:.2f}%"
)


# 11. CUSTOM PREDICTION

sample_flower = [[
    5.1,   # sepal length
    3.5,   # sepal width
    1.4,   # petal length
    0.2    # petal width
]]

prediction = logistic_model.predict(sample_flower)

predicted_species = iris.target_names[prediction[0]]

print("\n================================")
print("CUSTOM PREDICTION")

print("Predicted flower:", predicted_species)


# 12. SAVE TRAINED MODEL

joblib.dump(
    logistic_model,
    "iris_model.pkl"
)

print("\nModel saved as iris_model.pkl")

print("\n================================")
print("PROJECT COMPLETED")
print("================================")
