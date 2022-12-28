#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model


# In[10]:


df = pd.read_csv("homepricesm.csv")
df


# In[12]:


med = df.bedrooms.mean()
med


# In[13]:


df.bedrooms = df.bedrooms.fillna(med)
df


# In[14]:


reg = linear_model.LinearRegression()
reg.fit(df[["area","bedrooms","age"]], df.price)


# In[15]:


reg.coef_


# In[16]:


reg.intercept_


# In[17]:


reg.predict([[2000,4,20]])


# In[18]:


116.66950551*2000+18756.28806982*4+ (-3675.75111708*20) + 231586.00639409176
#X1 * AREA + X2*BEDROOMS +X3* AGE + B


# In[ ]:




