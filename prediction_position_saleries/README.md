# Regression Models Comparison Project

This project implements and compares various regression techniques to predict salaries based on position levels. It serves as a comprehensive study of different regression algorithms applied to the same dataset.

## Implemented Models

1. Support Vector Regression (SVR)
2. Decision Tree Regression
3. Polynomial Regression
4. Random Forest Regression

## Features

- Data preprocessing and feature scaling
- Implementation of multiple regression techniques
- Visualization of regression results
- Comparison of model performances

## Technologies Used

- Python 3.x
- NumPy
- Matplotlib
- Pandas
- scikit-learn

## Dataset

The project uses the `Position_Salaries.csv` file, which should contain the following columns:
- Position (categorical)
- Level (numerical)
- Salary (target variable)

Ensure this file is present in the project directory before running the scripts.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/regression-models-comparison.git
   cd regression-models-comparison
   ```

2. Install the required packages:
   ```
   pip install numpy matplotlib pandas scikit-learn
   ```

## Usage

Each regression model has its own Python script. To run a specific model, use:

```
python <model_name>.py
```

Replace `<model_name>` with one of the following:
- `svr_regression.py`
- `decision_tree_regression.py`
- `polynomial_regression.py`
- `random_forest_regression.py`

## Model Descriptions

### 1. Support Vector Regression (SVR)

- File: `svr_regression.py`
- Features:
  - Uses RBF (Radial Basis Function) kernel
  - Applies feature scaling
  - Visualizes results with scatter plot and prediction line

### 2. Decision Tree Regression

- File: `decision_tree_regression.py`
- Features:
  - Non-linear and non-continuous regression model
  - Does not require feature scaling
  - Visualizes results (implementation details in the script)

### 3. Polynomial Regression

- File: `polynomial_regression.py`
- Features:
  - Extends linear regression to polynomial relationships
  - Compares different degree polynomials
  - Visualizes results for each polynomial degree

### 4. Random Forest Regression

- File: `random_forest_regression.py`
- Features:
  - Ensemble learning method based on decision trees
  - Provides feature importance
  - Visualizes results (implementation details in the script)

## Visualizations

Each script generates visualizations to help understand the model's performance:
1. Scatter plot of the original data points
2. Line plot of the model's predictions
3. Higher resolution plot for smoother visualization (where applicable)

## Model Comparison

To compare the performance of different models:
1. Run each model script
2. Compare the visualizations and any printed metrics
3. Consider factors like complexity, interpretability, and accuracy for your specific use case

## Future Improvements

- Implement cross-validation for more robust model evaluation
- Add more regression techniques (e.g., Lasso, Ridge regression)
- Create a unified script to run and compare all models simultaneously
- Implement hyperparameter tuning for each model

## Contributing

Feel free to fork this project and submit pull requests with any enhancements. Areas for contribution include:
- Adding new regression techniques
- Improving visualization methods
- Implementing model comparison metrics

## License

This project is open source and available under the [MIT License](LICENSE).
