#!/usr/bin/env python
# coding: utf-8

# In[34]:


from sklearn.datasets import make_regression


# In[35]:


X,y = make_regression(n_samples=1000, n_features=2, noise=10, random_state=42)


# In[36]:


X.shape


# In[37]:


y.shape


# In[38]:


from sklearn.model_selection import train_test_split


# In[39]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.33, random_state=42)


# In[40]:


from sklearn.neighbors import KNeighborsRegressor


# In[41]:


print()


# In[42]:


regressor = KNeighborsRegressor(n_neighbors=6 , algorithm="auto")


# In[43]:


regressor.fit(X_train, y_train)


# In[44]:


y_predict=regressor.predict(X_test)


# In[45]:


y_predict


# In[46]:


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# In[47]:


print(r2_score(y_test,y_predict))


# In[48]:


print(mean_absolute_error(y_test,y_predict))


# In[49]:


print(mean_squared_error(y_test,y_predict))

