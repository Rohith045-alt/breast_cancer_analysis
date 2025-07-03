## Breast Cancer Classification using SVM (Scikit-learn)

This project demonstrates how to train and evaluate Support Vector Machine (SVM) models using the Breast Cancer Wisconsin dataset. It covers training with both linear and RBF kernels, visualization, hyperparameter tuning, and cross-validation.

##  Dataset

* `breast-cancer.csv` should be placed in the root directory.
* Binary target variable: `diagnosis` (M = malignant, B = benign)
* Two features used for visualization: `radius_mean`, `texture_mean`

##  Steps Covered

1. **Data Loading & Preprocessing**

   * Load data using pandas
   * Convert target labels (M/B to 1/0)
   * Feature scaling using `StandardScaler`

2. **Model Training**

   * Train SVM models with linear and RBF kernels using `SVC`

3. **Decision Boundary Visualization**

   * Plot decision boundaries for 2D feature space

4. **Hyperparameter Tuning**

   * Use `GridSearchCV` to tune `C` and `gamma` for RBF kernel

5. **Model Evaluation**

   * Evaluate both models using 5-fold cross-validation (`cv=5`)


##  Notes

* Only two features (`radius_mean` and `texture_mean`) are used for 2D visualization.
* You can modify the script to use all features for higher-dimensional accuracy.

##  Cross-validation (`cv=5`)

The dataset is split into 5 parts. The model trains on 4 parts and validates on 1, repeating this 5 times. This gives a more stable and fair evaluation.

##  Example Output

* Best parameters from Grid Search
* Accuracy from each fold of cross-validation
* Visual decision boundary plots for both kernels

---
