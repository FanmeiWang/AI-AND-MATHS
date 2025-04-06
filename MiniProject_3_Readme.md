# MiniProject 3: kNN, Decision Trees, and Conceptual Comparison

This repository contains a mini-project with **three parts** showcasing how mathematics and machine learning intersect in practice. 

- **Part (a)**: Implementing a **k-Nearest Neighbors (kNN)** classifier with:
  - \(k = 3\)
  - **Manhattan distance** plus **log transform** on the `Loan` feature
- **Part (b)**: Training a **Decision Tree** using **entropy** (information gain) as the splitting criterion
- **Part (c)**: A conceptual question comparing **kNN** and **Decision Trees** when a dataset has many irrelevant features

Below is a consolidated Python code snippet demonstrating parts (a) and (b). For **Part (c)**, a brief explanation is included at the end.

---

## 1. Code Implementation

You can run this code in a Jupyter Notebook or in a Python script (after installing `numpy`, `pandas`, `matplotlib`, and `scikit-learn`).

```python
import math
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# -----------------------------
# Part (a): kNN Implementation
# -----------------------------

# 1) Build the training dataset
data = {
    'Customer': ['A', 'B', 'C', 'D'],
    'Age':       [35,   52,   40,   60],
    'Loan':      [120000, 18000, 62000, 100000],
    'Default':   ['N',   'N',   'Y',   'Y']
}
df = pd.DataFrame(data)
print("Training Dataset:")
print(df)

# Define the new sample to predict
test_sample = {'Customer': 'E', 'Age': 48, 'Loan': 148000}

def manhattan_distance_with_log(age1, loan1, age2, loan2):
    """
    Distance formula:
    distance = |age1 - age2| + |log(loan1) - log(loan2)|
    Using natural log (math.log).
    """
    return abs(age1 - age2) + abs(math.log(loan1) - math.log(loan2))

def knn_predict(df, test_age, test_loan, k=3):
    # Calculate distance from the test sample to each training sample
    distances = []
    for i, row in df.iterrows():
        age_i  = row['Age']
        loan_i = row['Loan']
        label  = row['Default']
        
        dist = manhattan_distance_with_log(test_age, test_loan, age_i, loan_i)
        distances.append((label, dist))
    
    # Sort by distance ascending
    distances.sort(key=lambda x: x[1])
    
    # Get the top k neighbors
    neighbors = distances[:k]
    
    # Majority vote
    count_N = sum(1 for lbl, d in neighbors if lbl == 'N')
    count_Y = sum(1 for lbl, d in neighbors if lbl == 'Y')
    
    if count_N >= count_Y:
        prediction = 'N'
    else:
        prediction = 'Y'
    
    return prediction, neighbors

# Run KNN and display the result
knn_label, knn_neighbors = knn_predict(df, test_sample['Age'], test_sample['Loan'], k=3)
print("\n--- KNN (K=3) Prediction ---")
print("Test Sample = E (Age=48, Loan=148000)")
print("Neighbors (label, distance) =", knn_neighbors)
print("=> KNN Predicted Default =", knn_label)

# -----------------------------
# Part (b): Decision Tree
# -----------------------------

# Features: [Age, Loan], Target: [Default]
X = df[['Age', 'Loan']]
y = df['Default']

# Encode 'N'/'Y' into numeric form for the decision tree
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # e.g. 'N' -> 0, 'Y' -> 1

# Initialize a Decision Tree with entropy
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X, y_encoded)

# Predict with the Decision Tree
test_X = np.array([[test_sample['Age'], test_sample['Loan']]])
dt_pred_num = clf.predict(test_X)[0]
dt_pred_label = label_encoder.inverse_transform([dt_pred_num])[0]

print("\n--- Decision Tree Prediction ---")
print("Test Sample = E (Age=48, Loan=148000)")
print("=> Decision Tree Predicted Default =", dt_pred_label)

# Optional: visualize the decision tree
plt.figure(figsize=(8, 6))
plot_tree(
    clf,
    feature_names=['Age','Loan'],
    class_names=label_encoder.classes_,
    filled=True,
    rounded=True
)
plt.title("Decision Tree (Entropy)")
plt.show()
