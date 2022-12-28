#!/usr/bin/env python
# coding: utf-8

# In[4]:


pip install sklearn


# In[6]:


pip install pandas


# In[1]:



import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
import pandas as pd


# In[2]:


df = pd.read_csv("homeprices.csv")
df


# In[9]:


plt.xlabel("area(sqr ft)")
plt.ylabel("price(US$)")
plt.scatter(df.area, df.price)


# In[5]:


reg = linear_model.LinearRegression()
reg.fit(df[["area"]],df.price)


# In[7]:


reg.predict([[8000]])


# In[8]:


d = pd.read_csv("areas.csv")
d


# In[9]:


reg.predict(d)


# In[10]:


p = reg.predict(d)
d["prices"] = p


# In[11]:


d


# In[23]:


d.to_csv("predicton.csv")


# In[ ]:




