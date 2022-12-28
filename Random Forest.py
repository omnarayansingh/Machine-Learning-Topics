#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
from sklearn.datasets import load_digits
digits = load_digits()


# In[12]:


dir(digits)


# In[13]:


import matplotlib.pyplot as plt
plt.gray()
for i in range(4):
    plt.matshow(digits.images[i])


# In[14]:


df = pd.DataFrame(digits.data)
df.head()


# In[19]:


df["targets"] = digits.target
df.head()


# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(['targets'], axis = "columns"), digits.target,test_size = 0.2)


# In[21]:


len(X_test)


# In[18]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)


# In[9]:


model.score(X_test, y_test)


# In[10]:


X_test


# In[ ]:





# In[ ]:




