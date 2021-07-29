#SimpleLinear Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
X= dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values




#spliltting dataset 
from sklearn.model_selection import train_test_split 
X_train , X_test ,Y_train , Y_test = train_test_split(X, Y, test_size= 1/3, random_state= 0)

#Feature scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''
#fitting Simple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
#predicting the Test set results
y_pred = regressor.predict(X_test)

#visualising the training set results
plt.scatter(X_train, Y_train , color = 'red')
plt.plot(X_train, regressor.predict(X_train),color = 'blue' )
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
#visualising the test set results
plt.scatter(X_test, Y_test , color = 'red')
plt.plot(X_train, regressor.predict(X_train),color = 'blue' )
plt.title('Salary vs Experience (test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()






                