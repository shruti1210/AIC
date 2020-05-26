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

from sklearn.svm import SVR #Importing SVR function for support vector regression
svreg = SVR(kernel='linear')
svreg.fit(X_train, y_train) #fitting training data on SVR model

y_pred = svreg.predict(X_test) #obtaining predictions on test data

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)
from sklearn import metrics  #testing the model's performance

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) #MAE 
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  #MSE
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) #RMSE











