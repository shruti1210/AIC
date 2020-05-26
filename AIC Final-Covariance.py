#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import sys
import array 


# In[2]:


df_data = pd.read_csv("BITS AIC 2019 - Reflexis Raw Dataset (1).csv")


# In[4]:


df_cs = pd.read_csv("wow.csv")  


# In[20]:


#result_array = np.array([])
w, h = len(df_cs), 2;
Matrix = [[0 for x in range(w)] for y in range(h)] 

#print(df_data[0:df_cs['Cummulative count'][0]-1].corr())

for i in range(len(df_cs)-1):
    m = df_cs['Cummulative count'][i]
    k = df_cs['Cummulative count'][i+1]
    a= df_cs['Store'][i]
    
    #print(i)    
    print(df_data[m:k-1].corr())
    df_correlation = df_data[m:k-1].corr()
    result = df_correlation['MANAGER_SCHED_HOURS']['EFFECTIVITY']
    #result = do_stuff( df_correlation['MANAGER_SCHED_HOURS']['P1'])
   # Matrix[][] = np.insert(Matrix, result)
    Matrix[1][i]=result
    Matrix[0][i]=a
    
print(df_data[m+1:len(df_data)].corr())#printing the last slot as above i+1 gives error 
result = df_correlation['MANAGER_SCHED_HOURS']['EFFECTIVITY']
Matrix[1][i]=result
Matrix[0][i]= df_cs['Store'][i] #separately for the last slot 


# In[21]:


import numpy as np

Mat1 = np.asarray(Matrix)


# In[23]:


matplotlib.rcParams.update({'font.size': 14})

f, axarr = plt.subplots(1,2, figsize=(20, 4))
axarr[0].scatter(Mat1[0,:], Mat1[1,:],
                 edgecolor='black', linewidth='1', s=70, alpha=0.7, c="#e84629")
axarr[0].set_xlabel("Store Number")
axarr[0].set_ylabel("correlation of effectivity and MANAGER_SCHED_HOURS")
axarr[0].set_ylim(0, 1)
axarr[0].set_yticks(np.arange(-.5, 1, .1))
axarr[0].set_xticks(np.arange(0, 8000, 1000))
axarr[0].grid(color='red', linestyle='--', linewidth=1, alpha=0.2)
axarr[0].spines["top"].set_visible(False)
axarr[0].spines["right"].set_visible(False)
axarr[0].spines["bottom"].set_visible(False)
axarr[0].spines["left"].set_visible(False)


# In[24]:


m1=max(Mat1[1])
m1


# In[25]:


#result_array = np.array([])
w, h = len(df_cs), 2;
Matrix = [[0 for x in range(w)] for y in range(h)] 

#print(df_data[0:df_cs['Cummulative count'][0]-1].corr())

for i in range(len(df_cs)-1):
    m = df_cs['Cummulative count'][i]
    k = df_cs['Cummulative count'][i+1]
    a= df_cs['Store'][i]
    
    #print(i)    
    print(df_data[m:k-1].corr())
    df_correlation = df_data[m:k-1].corr()
    result = df_correlation['MANAGER_SCHED_HOURS']['Average Sale Purchase']
    #result = do_stuff( df_correlation['MANAGER_SCHED_HOURS']['P1'])
   # Matrix[][] = np.insert(Matrix, result)
    Matrix[1][i]=result
    Matrix[0][i]=a
    
print(df_data[m+1:len(df_data)].corr())#printing the last slot as above i+1 gives error 
result1 = df_correlation['MANAGER_SCHED_HOURS']['Average Sale Purchase']
Matrix[1][i]=result
Matrix[0][i]= df_cs['Store'][i] #separately for the last slot 


# In[26]:


Mat2 = np.asarray(Matrix)


# In[27]:


matplotlib.rcParams.update({'font.size': 14})

f, axarr = plt.subplots(1,2, figsize=(20, 4))
axarr[0].scatter(Mat2[0,:], Mat2[1,:],
                 edgecolor='black', linewidth='1', s=70, alpha=0.7, c="#e84629")
axarr[0].set_xlabel("Store Number")
axarr[0].set_ylabel("correlation of average store sales and MANAGER_SCHED_HOURS")
axarr[0].set_ylim(0, 1)
axarr[0].set_yticks(np.arange(-.5, 1, .2))
axarr[0].set_xticks(np.arange(0, 8000, 1000))
axarr[0].grid(color='red', linestyle='--', linewidth=1, alpha=0.2)
axarr[0].spines["top"].set_visible(False)
axarr[0].spines["right"].set_visible(False)
axarr[0].spines["bottom"].set_visible(False)
axarr[0].spines["left"].set_visible(False)


# In[28]:


m2=max(Mat2[1])
m2


# In[30]:


#result_array = np.array([])
w, h = len(df_cs), 2;
Matrix = [[0 for x in range(w)] for y in range(h)] 

#print(df_data[0:df_cs['Cummulative count'][0]-1].corr())

for i in range(len(df_cs)-1):
    m = df_cs['Cummulative count'][i]
    k = df_cs['Cummulative count'][i+1]
    a= df_cs['Store'][i]
    
    #print(i)    
    print(df_data[m:k-1].corr())
    df_correlation = df_data[m:k-1].corr()
    result = df_correlation['CHANGE']['EFFECTIVITY']
    #result = do_stuff( df_correlation['MANAGER_SCHED_HOURS']['P1'])
   # Matrix[][] = np.insert(Matrix, result)
    Matrix[1][i]=result
    Matrix[0][i]=a
    
print(df_data[m+1:len(df_data)].corr())#printing the last slot as above i+1 gives error 
result1 = df_correlation['CHANGE']['EFFECTIVITY']
Matrix[1][i]=result
Matrix[0][i]= df_cs['Store'][i] #separately for the last slot 


# In[31]:



Mat3 = np.asarray(Matrix)


# In[32]:


matplotlib.rcParams.update({'font.size': 14})

f, axarr = plt.subplots(1,2, figsize=(20, 4))
axarr[0].scatter(Mat3[0,:], Mat3[1,:],
                 edgecolor='black', linewidth='1', s=70, alpha=0.7, c="#e84629")
axarr[0].set_xlabel("Store Number")
axarr[0].set_ylabel("CHANGE and EFFECTIVITY")
axarr[0].set_ylim(0, 1)
axarr[0].set_yticks(np.arange(-.5, 1, .1))
axarr[0].set_xticks(np.arange(0, 8000, 1000))
axarr[0].grid(color='red', linestyle='--', linewidth=1, alpha=0.2)
axarr[0].spines["top"].set_visible(False)
axarr[0].spines["right"].set_visible(False)
axarr[0].spines["bottom"].set_visible(False)
axarr[0].spines["left"].set_visible(False)


# In[33]:


m3=max(Mat3[1])
print (m3)

plt.show()
# In[ ]:




