#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
df = pd.read_csv("salaries.csv")
df.head()


# In[3]:


inputs = df.drop("salary_more_then_100k", axis = "columns")
targets = df["salary_more_then_100k"]


# In[5]:


targets


# In[6]:


from sklearn.preprocessing import LabelEncoder


# In[7]:


le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()


# In[8]:


inputs["company_n"] = le_company.fit_transform(inputs["company"])
inputs["job_n"] = le_job.fit_transform(inputs["job"])
inputs["degree_n"] = le_degree.fit_transform(inputs["degree"])
inputs.head()


# In[12]:


inputs_n = inputs.drop(["job","degree", "company"], axis = "columns")
inputs_n


# In[9]:


from sklearn import tree


# In[10]:


model = tree.DecisionTreeClassifier()


# In[13]:


model.fit(inputs_n,targets)


# models.score(inputs_n, targets)

# In[14]:


model.score(inputs_n,targets)


# In[15]:


model.predict([[1,2,1]])


# In[ ]:




