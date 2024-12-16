#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import load_iris


# In[3]:


from sklearn.model_selection import train_test_split


# In[9]:


X, y = load_iris( return_X_y=True)


# In[10]:


X


# In[11]:


y


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=48)


# In[13]:


from sklearn.naive_bayes import GaussianNB


# In[16]:


gnb = GaussianNB()


# In[18]:


gnb.fit(X_train,y_train)


# In[ ]:


gnb.fit(X_train,y_train)


# In[19]:


y_predict= gnb.predict(X_test)


# In[20]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[21]:


print(confusion_matrix(y_predict, y_test))


# In[23]:


print(accuracy_score(y_predict, y_test))


# In[22]:


print(classification_report(y_predict, y_test))

