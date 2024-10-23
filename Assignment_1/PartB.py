import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from IPython.display import *
from matplotlib.colors import *
from sklearn.datasets import *
from sklearn.model_selection import *
from sklearn.neighbors import *
from sklearn.naive_bayes import *
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.metrics import *

# Task B.1
# Generating an artificial dataset
X_class, y_class = make_classification(
    n_samples = 80, # Total of 80 data points
    n_features = 2, # 2 continuous independent attributes
    n_informative = 2, # Both attributes are informative
    n_redundant = 0, # No redundant features
    n_clusters_per_class = 1, # 
    n_classes = 2, # Binary class
    random_state = 42 # 
)

# Task B.2
# Splitting the data into training and test sets (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Add outliers (2 per class) to the training set
np.random.seed(42)
outliers_class0 = np.random.uniform(low=-5, high=5, size=(2, 2))
outliers_class1 = np.random.uniform(low=-5, high=5, size=(2, 2))
X_train = np.vstack([X_train, outliers_class0, outliers_class1])
y_train = np.hstack([y_train, [0, 0, 1, 1]])

# Plot the generated dataset with outliers, and display the the shapes of the training and data sets
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', edgecolors='k', label='Training Data')
plt.title('Artificial Binary Classification Dataset With Outliers')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Task B.3
# Initializing and train each of the classifiers
knn = KNeighborsClassifier()
nb = GaussianNB()
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(random_state=42)

knn.fit(X_train, y_train)
nb.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Making predictions on the data set
y_pred_knn = knn.predict(X_test)
y_pred_nb = nb.predict(X_test)
y_pred_dt = dt.predict(X_test)
y_pred_rf = rf.predict(X_test)

# Calculate training and test accuracy for each model
train_acc_knn = accuracy_score(y_train, knn.predict(X_train))
test_acc_knn = accuracy_score(y_test, y_pred_knn)
train_acc_nb = accuracy_score(y_train, nb.predict(X_train))
test_acc_nb = accuracy_score(y_test, y_pred_nb)
train_acc_dt = accuracy_score(y_train, dt.predict(X_train))
test_acc_dt = accuracy_score(y_test, y_pred_dt)
train_acc_rf = accuracy_score(y_train, rf.predict(X_train))
test_acc_rf = accuracy_score(y_test, y_pred_rf)

# Create confusion matrices for each model
confusion_knn = confusion_matrix(y_test, y_pred_knn)
confusion_nb = confusion_matrix(y_test, y_pred_nb)
confusion_dt = confusion_matrix(y_test, y_pred_dt)
confusion_rf = confusion_matrix(y_test, y_pred_rf)

# Organization of results
results = {
    "Model": ['k-NN', 'Naive Bayes', 'Decision Tree', 'Random Forest'],
    "Train Accuracy": [train_acc_knn, train_acc_nb, train_acc_dt, train_acc_rf],
    "Test Accuracy": [test_acc_knn, test_acc_nb, test_acc_dt, test_acc_rf],
    "Confusion Matrix": [confusion_knn, confusion_nb, confusion_dt, confusion_rf]
}

# Conversion to DataFrame, for better visualization
results_df = pd.DataFrame(results)
display(results)

# Function to plot decision boundaries
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(('blue', 'red')))
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=ListedColormap(('blue', 'red')))
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)

# Plot decision boundaries for each model
plt.figure(figsize=(8, 8))

plt.subplot(2, 2, 1)
plot_decision_boundary(knn, X_train, y_train, 'k-NN Decision Boundary')

plt.subplot(2, 2, 2)
plot_decision_boundary(nb, X_train, y_train, 'Naive Bayes Decision Boundary')

plt.subplot(2, 2, 3)
plot_decision_boundary(dt, X_train, y_train, 'Decision Tree Decision Boundary')

plt.subplot(2, 2, 4)
plot_decision_boundary(rf, X_train, y_train, 'Random Tree Forest Decision Boundary')

plt.tight_layout()
plt.show()