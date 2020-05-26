import pandas as pd
import random
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

dataset = pd.read_csv('file1.csv')
print (dataset.shape)
print (dataset.head())

X1=dataset.drop('DATE',axis=1)
X=X1.drop('SYSTEM_SCHED_HOURS',axis=1)
y=dataset['SYSTEM_SCHED_HOURS']
#print (X.head())

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state=42)


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train)
print ("=====LINEAR REGRESSION=====")

y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)  
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#print (X_train.head())
poly = PolynomialFeatures(degree = 4) 
X_poly = poly.fit_transform(X_train)
X_poly1 = poly.fit_transform(X_test)
#print(X_poly.head())
poly.fit(X_poly, y_train)
print ("=====POLYNOMIAL REGRESSION=====")

lin2 = LinearRegression() 
lin2.fit(X_poly, y_train)
y_pred = lin2.predict(X_poly1)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)  
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#print (X_train.head())
