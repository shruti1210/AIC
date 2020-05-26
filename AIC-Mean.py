#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import sys
import array 


# In[5]:


df_data = pd.read_csv("BITS AIC 2019 - Reflexis Raw Dataset (1).csv")


# In[6]:


df_cs = pd.read_csv("wow.csv")  


# In[19]:



# In[13]:


#df_mean = df_data[1:757].mean()
list=[]
store=[]
#print(df_data[0:df_cs['Cummulative count'][0]-1].corr())

for i in range(len(df_cs)-1):
    if(i==0):
        df_mean = df_data[0:756].mean()
        list.append(df_mean['Average Sale Purchase'])
        store.append(df_mean['STORE'])
        

    else:
        m = df_cs['Cummulative count'][i]
        k = df_cs['Cummulative count'][i+1]
        a= df_cs['Store'][i]
        df_mean = df_data[m:k-1].mean()
        list.append(df_mean['Average Sale Purchase'])
        store.append(df_mean['STORE'])
        #print(df_mean)
#print(list)  
#list.sort()
print(list)    
print(store)


# In[18]:


matplotlib.rcParams.update({'font.size': 14})

f, axarr = plt.subplots(1,2, figsize=(20, 4))
axarr[0].scatter( store, list,
                 edgecolor='black', linewidth='1', s=70, alpha=0.7, c="#e84629")
axarr[0].set_xlabel("Store Number")
axarr[0].set_ylabel("av of average sales")
axarr[0].set_ylim(0, 1)
axarr[0].set_yticks(np.arange(30, 80, 10))
axarr[0].set_xticks(np.arange(0, 8000, 1000))
axarr[0].grid(color='red', linestyle='--', linewidth=1, alpha=0.2)
axarr[0].spines["top"].set_visible(False)
axarr[0].spines["right"].set_visible(False)
axarr[0].spines["bottom"].set_visible(False)
axarr[0].spines["left"].set_visible(False)
plt.show()
