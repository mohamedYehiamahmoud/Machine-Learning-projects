# Salary Prediction Project

This project implements a simple linear regression model to predict salaries based on years of experience. It uses Python and popular data science libraries to analyze the relationship between work experience and salary.

## Features

- Data loading and preprocessing
- Simple linear regression model implementation
- Model training and prediction
- Visualization of training and test results

## Technologies Used

- Python 3.x
- NumPy
- Matplotlib
- Pandas
- scikit-learn

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/salary-prediction-project.git
   cd salary-prediction-project
   ```

2. Install the required packages:
   ```
   pip install numpy matplotlib pandas scikit-learn
   ```

3. Ensure you have the `Salary_Data.csv` file in the project directory.

## Usage

Run the `simple_linear_regression.py` script:

```
python simple_linear_regression.py
```

This will:
1. Load and preprocess the data
2. Split the dataset into training and test sets
3. Train a simple linear regression model
4. Make predictions on the test set
5. Visualize the results for both training and test sets

## Data

The project uses the `Salary_Data.csv` file, which should contain two columns:
- Years of Experience
- Salary

Ensure this file is present in the project directory before running the script.

## Visualizations

The script generates two plots:
1. Salary vs Experience (Training Set)
2. Salary vs Experience (Test Set)

These visualizations help in understanding the relationship between years of experience and salary, as well as the model's performance.

## Notes

- The script currently uses a 1/3 split for the test set. You can adjust this in the `train_test_split` function call.
- Feature scaling is commented out in the current version. Uncomment and modify if needed for your specific dataset.

## Future Improvements

- Implement more advanced regression techniques
- Add error metrics for model evaluation
- Create a user interface for easy prediction input

## Contributing

Feel free to fork this project and submit pull requests with any enhancements.

## License

This project is open source and available under the [MIT License](LICENSE).
