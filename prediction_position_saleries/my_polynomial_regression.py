# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""
#Fittng linear  regression to the dataset
from sklearn.linear_model import LinearRegression
Lin_reg =LinearRegression()
Lin_reg.fit(X,y)
#Fittng polnomial  regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree= 4)
X_poly = poly_reg.fit_transform(X)
Lin_reg2 = LinearRegression()
Lin_reg2.fit(X_poly,y)

#visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, Lin_reg.predict(X), color= 'blue')
plt.title('Truth or Bluff(LinearRegression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
#visualising the poly Regression results
X_grid =np.arange(min(X),max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, Lin_reg2.predict(poly_reg.fit_transform(X_grid)), color= 'blue')
plt.title('Truth or Bluff(PolynomialRegression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
#predicting new results with linear Regression
Lin_reg.predict([[6.5]])
#predicting new results with poly Regression
Lin_reg2.predict(poly_reg.fit_transform([[6.5]]))