Simple Linear Regression: Salary Prediction
This project implements a simple linear regression model to predict salaries based on years of experience using the Salary_Data.csv dataset.

Project Structure
mysimple_linear_regression.py: The main Python script for building and visualizing the regression model.
Salary_Data.csv: The dataset containing information on employees' years of experience and their corresponding salaries.
Requirements
Install necessary Python libraries using:

bash
Copy code
pip install numpy pandas matplotlib scikit-learn
Workflow
Data Preprocessing:

Load the dataset using Pandas.
Split data into training and test sets using train_test_split from Scikit-learn.
Model Training:

Use the LinearRegression model from Scikit-learn to fit the training data.
Prediction:

Predict salaries for the test set using the trained model.
Visualization:

Visualize training and test results with Matplotlib scatter plots.
Visualizations
Training Set: Red points represent actual data, and the blue line represents the regression fit.
Test Set: Compares the predicted vs actual salaries.
License
This project is open-source and available under the MIT License.
