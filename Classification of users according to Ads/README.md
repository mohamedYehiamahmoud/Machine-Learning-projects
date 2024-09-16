
# Machine Learning Classification Project

This project implements multiple classification algorithms on the **Social_Network_Ads** dataset to compare their performances. Each classifier is implemented in a separate Python script and is used to classify social network ads based on age and estimated salary.

## Project Structure

The repository contains the following files:

- `my_logistic_regression.py` - Logistic Regression implementation
- `my_Knn.py` - K-Nearest Neighbors (KNN) implementation
- `MY_SVM.py` - Support Vector Machine (SVM) implementation
- `mykernel_svm.py` - Kernel SVM implementation
- `my_naive_bayes.py` - Naive Bayes implementation
- `my_decision_tree_classification.py` - Decision Tree implementation
- `my_random_forest_classification.py` - Random Forest implementation

### Dataset

The dataset used is `Social_Network_Ads.csv`, which contains information about users' ages, estimated salaries, and whether they purchased a particular product from the social network ads.

The features used in classification are:
- **Age**
- **Estimated Salary**

The target variable is:
- **Purchased** (whether the user purchased the product or not)

### Installation and Requirements

To run the scripts, you'll need to install the following Python libraries:

```bash
pip install numpy pandas matplotlib scikit-learn
```

### Usage

Each script follows the same general structure:

1. **Importing Libraries**: The necessary libraries are imported, such as NumPy, Pandas, Matplotlib, and Scikit-learn.
2. **Loading the Dataset**: The `Social_Network_Ads.csv` dataset is loaded into the script.
3. **Data Preprocessing**:
   - The dataset is split into training and test sets using `train_test_split`.
   - Feature scaling is applied to standardize the data.
4. **Training the Model**: The classifier is trained using the training data.
5. **Prediction**: The model predicts the target variable for the test data.
6. **Evaluation**: A confusion matrix is used to evaluate the performance of the classifier.
7. **Visualization**: The training and test results are visualized using a scatter plot.

### Example: Decision Tree Classification

The `my_decision_tree_classification.py` file contains an implementation of the Decision Tree classifier.

**Steps**:
1. Import the required libraries (`numpy`, `matplotlib`, `pandas`, and `sklearn`).
2. Load the dataset from `Social_Network_Ads.csv`.
3. Split the dataset into training and test sets.
4. Apply feature scaling using `StandardScaler`.
5. Fit a `DecisionTreeClassifier` to the training data.
6. Predict the results for the test data.
7. Use a confusion matrix for performance evaluation.
8. Visualize both the training and test set results using Matplotlib.

```python
# Example snippet from my_decision_tree_classification.py

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Evaluating with confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
```

The training and test results are visualized using scatter plots with distinct colors for each class.

### Visualizations

Each script includes visualization of:
- The decision boundary for both training and test sets.
- Scatter plots showing how well the classifier differentiates between different classes.

### License

This project is open-source and available under the MIT License.
