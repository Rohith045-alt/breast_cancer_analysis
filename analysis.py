import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
df = pd.read_csv('breast-cancer.csv')
df.head()
df.info()
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scale = StandardScaler()
X_train_scaled = scale.fit_transform(X_train)
X_test_scaled = scale.transform(X_test)
svm_linear = SVC(kernel = 'linear', C=1.0)
svm_linear.fit(X_train_scaled, y_train)
accuracy = svm_linear.score(X_test_scaled, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")
svm_rbf = SVC(kernel = 'rbf', C=1.0)
svm_rbf.fit(X_train_scaled, y_train)
rbf_accuracy = svm_rbf.score(X_test_scaled, y_test)
print(f"RBF Model accuracy: {rbf_accuracy * 100:.2f}%)")
X = df[['radius_mean', 'texture_mean']]
y = df['diagnosis'].map({'M': 1, 'B': 0})  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train_scaled, y_train)

svm_rbf = SVC(kernel='rbf', C=1.0)
svm_rbf.fit(X_train_scaled, y_train)

X_combined = np.vstack((X_train_scaled, X_test_scaled))
y_combined = np.hstack((y_train, y_test))

def plot_decision_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel("radius_mean")
    plt.ylabel("texture_mean")
    plt.title(title)
    plt.show()

plot_decision_boundary(svm_linear, X_combined, y_combined, "SVM with Linear Kernel")
plot_decision_boundary(svm_rbf, X_combined, y_combined, "SVM with RBF Kernel")
X = df[['radius_mean', 'texture_mean']]
y = df['diagnosis'].map({'M': 1, 'B': 0})
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

svm = SVC(kernel='rbf')

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
}
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_scaled, y)
print("Best parameters:", grid_search.best_params_)
print(f"Best cross-validation accuracy: {grid_search.best_score_ * 100:.2f}%")
cv_scores_linear = cross_val_score(svm_linear, X_scaled, y, cv=5)
cv_scores_rbf = cross_val_score(svm_rbf, X_scaled, y, cv=5)

print("Linear SVM - Cross-validation accuracy scores:", cv_scores_linear)
print(f"Linear SVM - Mean Accuracy: {cv_scores_linear.mean() * 100:.2f}%")

print("RBF SVM - Cross-validation accuracy scores:", cv_scores_rbf)
print(f"RBF SVM - Mean Accuracy: {cv_scores_rbf.mean() * 100:.2f}%")
