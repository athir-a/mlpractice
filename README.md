# mlpractice
# Machine Learning Lab — Ready-to-Run Python Programs

Each experiment has a separate, complete Python program. Copy the code block for the experiment you need.

## Experiment 1 — Linear Regression (Gradient Descent and Normal Equation)

**Install dependencies**
```bash
pip install numpy pandas matplotlib scikit-learn
```

**Code**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the dataset and select one feature.
data = fetch_california_housing(as_frame=True)
X = data.data[["AveRooms"]].to_numpy()
y = data.target.to_numpy()

# 2. Split the data before fitting the scaler to avoid data leakage.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).ravel()
X_test_scaled = scaler.transform(X_test).ravel()

# 3. Train using gradient descent.
def gradient_descent(X, y, learning_rate=0.1, epochs=2000):
    m = len(y)
    w0 = 0.0
    w1 = 0.0

    for _ in range(epochs):
        predictions = w0 + w1 * X
        errors = predictions - y

        grad_w0 = np.sum(errors) / m
        grad_w1 = np.dot(errors, X) / m

        w0 -= learning_rate * grad_w0
        w1 -= learning_rate * grad_w1

    return w0, w1

w0_gd, w1_gd = gradient_descent(
    X_train_scaled, y_train, learning_rate=0.1, epochs=2000
)
y_pred_gd = w0_gd + w1_gd * X_test_scaled

# 4. Train using the numerically safer least-squares solution.
X_train_bias = np.column_stack(
    (np.ones(X_train_scaled.shape[0]), X_train_scaled)
)
theta, *_ = np.linalg.lstsq(X_train_bias, y_train, rcond=None)

X_test_bias = np.column_stack(
    (np.ones(X_test_scaled.shape[0]), X_test_scaled)
)
y_pred_ne = X_test_bias @ theta

# 5. Evaluate both models.
for name, predictions in [
    ("Gradient Descent", y_pred_gd),
    ("Normal Equation / Least Squares", y_pred_ne),
]:
    print(f"\n{name}")
    print(f"MSE: {mean_squared_error(y_test, predictions):.4f}")
    print(f"R-squared: {r2_score(y_test, predictions):.4f}")

print("\nGradient Descent weights:", w0_gd, w1_gd)
print("Least-squares weights:", theta)

# 6. Plot training data and the gradient-descent regression line.
x_line = np.linspace(X_train_scaled.min(), X_train_scaled.max(), 200)
plt.scatter(X_train_scaled, y_train, s=8, alpha=0.35, label="Training data")
plt.plot(x_line, w0_gd + w1_gd * x_line, color="red", label="Gradient Descent fit")
plt.xlabel("AveRooms (standardized)")
plt.ylabel("Median House Value ($100,000s)")
plt.title("Linear Regression — California Housing")
plt.legend()
plt.tight_layout()
plt.show()
```

**Note:** The first run may download the California Housing dataset.

---

## Experiment 4 — Logistic Regression (MLE, L2-MAP and L1-MAP)

**Install dependencies**
```bash
pip install numpy scikit-learn
```

**Code**
```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# 1. Load and split the dataset.
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 2. Standardize using training data only.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. MLE: logistic regression without a regularization penalty.
mle_model = LogisticRegression(
    penalty=None,
    solver="lbfgs",
    max_iter=5000,
)
mle_model.fit(X_train_scaled, y_train)

# 4. MAP with an L2 penalty (Gaussian-prior interpretation).
map_l2_model = LogisticRegression(
    penalty="l2",
    C=1.0,
    solver="lbfgs",
    max_iter=5000,
)
map_l2_model.fit(X_train_scaled, y_train)

# 5. MAP with an L1 penalty (Laplace-prior interpretation).
map_l1_model = LogisticRegression(
    penalty="l1",
    C=1.0,
    solver="liblinear",
    max_iter=5000,
)
map_l1_model.fit(X_train_scaled, y_train)

# 6. Evaluate all three models.
def evaluate(model, name):
    predictions = model.predict(X_test_scaled)

    print(f"\n{name}")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_test, predictions, zero_division=0):.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print(f"Mean absolute coefficient: {np.mean(np.abs(model.coef_)):.6f}")
    print(f"Zero coefficients: {np.count_nonzero(model.coef_ == 0)}")

evaluate(mle_model, "MLE (No Regularization)")
evaluate(map_l2_model, "MAP with L2 Prior")
evaluate(map_l1_model, "MAP with L1 Prior")
```

**Note:** This uses scikit-learn's built-in Breast Cancer Wisconsin dataset; no separate dataset download is needed.

---

## Experiment 8 — K-Nearest Neighbors (Fashion MNIST)

**Install dependencies**
```bash
pip install numpy matplotlib scikit-learn tensorflow
```

**Code**
```python
import time
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load Fashion MNIST.
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# 2. Flatten each 28 x 28 image and normalize pixels to [0, 1].
X_train = X_train.reshape(len(X_train), -1).astype("float32") / 255.0
X_test = X_test.reshape(len(X_test), -1).astype("float32") / 255.0

# Use subsets to keep this KNN experiment manageable.
X_train_small = X_train[:5000]
y_train_small = y_train[:5000]
X_test_small = X_test[:1000]
y_test_small = y_test[:1000]

# 3. Compare different values of K.
k_values = [1, 3, 5, 7, 9, 15]
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)

    fit_start = time.perf_counter()
    model.fit(X_train_small, y_train_small)
    fit_seconds = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    predictions = model.predict(X_test_small)
    predict_seconds = time.perf_counter() - predict_start

    accuracy = accuracy_score(y_test_small, predictions)
    accuracies.append(accuracy)

    print(
        f"K={k:<2} Accuracy={accuracy:.4f} "
        f"Fit time={fit_seconds:.2f}s "
        f"Prediction time={predict_seconds:.2f}s"
    )

# 4. Plot accuracy against K.
plt.plot(k_values, accuracies, marker="o")
plt.xlabel("K (Number of Neighbors)")
plt.ylabel("Test Accuracy")
plt.title("KNN Accuracy vs K — Fashion MNIST")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Note:** The first run may download Fashion MNIST. KNN prediction can be slow because it compares test images with stored training examples.

---

## Experiment 11 — Linear SVM (Iris: Setosa vs Non-Setosa)

**Install dependencies**
```bash
pip install numpy matplotlib scikit-learn
```

**Code**
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Load Iris and select petal length and petal width.
iris = load_iris()
X = iris.data[:, [2, 3]]
y = (iris.target == 0).astype(int)  # 1 = Setosa, 0 = Non-Setosa

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)

# 2. Train the linear SVM.
svm = SVC(kernel="linear", C=1.0)
svm.fit(X_train, y_train)
predictions = svm.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print(f"Number of support vectors: {len(svm.support_vectors_)}")

# 3. Calculate the decision boundary and margin lines.
w = svm.coef_[0]
b = svm.intercept_[0]

x_plot = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200)
decision_boundary = -(w[0] * x_plot + b) / w[1]

# The margin lines satisfy w[0]*x + w[1]*y + b = +/-1.
margin_upper = (1 - b - w[0] * x_plot) / w[1]
margin_lower = (-1 - b - w[0] * x_plot) / w[1]

# 4. Plot both classes, the boundary, margins, and support vectors.
plt.scatter(
    X[y == 1, 0], X[y == 1, 1],
    label="Setosa", alpha=0.8
)
plt.scatter(
    X[y == 0, 0], X[y == 0, 1],
    label="Non-Setosa", alpha=0.8
)
plt.plot(x_plot, decision_boundary, "k-", label="Decision boundary")
plt.plot(x_plot, margin_upper, "k--", label="Margin")
plt.plot(x_plot, margin_lower, "k--")
plt.scatter(
    svm.support_vectors_[:, 0],
    svm.support_vectors_[:, 1],
    s=150,
    facecolors="none",
    edgecolors="red",
    label="Support vectors",
)
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Linear SVM — Iris (Setosa vs Non-Setosa)")
plt.legend()
plt.tight_layout()
plt.show()
```

**Note:** The margin lines are calculated from the SVM equations rather than approximated from the margin width.

---

## Experiment 13 — Multilayer Perceptron (Wine Quality)

**Install dependencies**
```bash
pip install numpy pandas matplotlib scikit-learn
```

**Code**
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the red Wine Quality dataset from the UCI repository.
url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)
df = pd.read_csv(url, sep=";")

X = df.drop(columns=["quality"])
y = df["quality"]

# 2. Split the data and standardize features.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Compare network architectures.
architectures = [(10,), (50,), (50, 30), (100, 50, 25)]
plt.figure(figsize=(9, 5))

for architecture in architectures:
    model = MLPRegressor(
        hidden_layer_sizes=architecture,
        max_iter=1000,
        random_state=42,
        early_stopping=True,
    )
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Architecture={architecture} | MSE={mse:.3f} | R2={r2:.3f}")

    plt.plot(model.loss_curve_, label=str(architecture))

# 4. Plot the training-loss curves.
plt.xlabel("Training iteration")
plt.ylabel("Training loss")
plt.title("MLP Training Loss — Red Wine Quality")
plt.legend(title="Hidden layers")
plt.tight_layout()
plt.show()
```

**Note:** This program downloads the red Wine Quality CSV from the UCI repository, so an internet connection is required.
