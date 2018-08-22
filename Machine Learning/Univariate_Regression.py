import numpy as np
import pandas as pd
import math
import matplotlib
#matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from pandas import DataFrame, Series
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original",
                   delim_whitespace = True, header=None,
                   names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration',
                            'model', 'origin', 'car_name'])

data = data.dropna()
data.head()

target_vars = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']
dep_vars = ['mpg']
target_data = data[target_vars]
dep_data = data[dep_vars]


data.to_csv('auto-mpg.csv', index=False)

target_train, target_test, dep_train, dep_test = train_test_split(target_data, dep_data, test_size=0.33, random_state=42)

# Train Model
regr = linear_model.LinearRegression()
regr.fit(target_train, dep_train)

#Coefficients
print('Coefficients: {0}'.format((target_vars,np.squeeze(regr.coef_))))


# Residual

regr_predict = regr.predict(target_test)
print("Residual sum of squares: %.2f"
      % np.mean((regr_predict - dep_test) ** 2))

# Intercepts
datamt = np.matrix(data)                                            
lmFit = regr.fit(datamt[:,3], datamt[:,1])                            
print('Intercepts\n', lmFit.coef_, '    ', lmFit.intercept_)


# Accuracy Check
regr = linear_model.LinearRegression()
regr.fit(target_train, dep_train)

y_predict = regr.predict(target_test)

regr_msq = mean_squared_error(y_predict, dep_test)

print('Accuracy Check:', regr_msq)

print('Sqr:', math.sqrt(regr_msq))


# Scatter plot
X = data['mpg']
Y = data['weight']

X=X.reshape(len(X),1)
Y=Y.reshape(len(Y),1)
 
# Split the data into training/testing sets
X_train = X[:-80]
X_test = X[-20:]
 
# Split the targets into training/testing sets
Y_train = Y[:-80]
Y_test = Y[-20:]
 
# Plot outputs
plt.scatter(X_test, Y_test,  color='black')
plt.title('Test Data')
plt.xlabel('Size')
plt.ylabel('Price')
plt.xticks(())
plt.yticks(())
 
plt.show()
