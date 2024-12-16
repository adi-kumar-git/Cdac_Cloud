#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


from sklearn.datasets import make_classification


# In[15]:


X,y=make_classification(n_samples=1000,n_features=3,n_redundant=1, n_classes=2, random_state=999)


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)


# In[18]:


from sklearn.neighbors import KNeighborsClassifier


# In[19]:


classifier =KNeighborsClassifier(n_neighbors=5, algorithm="auto")


# In[20]:


classifier.fit(X_train, y_train)


# In[22]:


y_predict = classifier.predict(X_test)


# In[23]:


from sklearn.metrics import confusion_matrix, accuracy_score,classification_report


# In[24]:


print(confusion_matrix(y_predict,y_test))


# In[25]:


print(accuracy_score(y_predict,y_test))


# In[26]:


print(classification_report(y_predict,y_test))

