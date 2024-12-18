#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[423]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[424]:


from sklearn.datasets import make_classification


# In[425]:


X,y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=98)


# In[426]:


pd.DataFrame(X)


# In[427]:


from sklearn.model_selection import train_test_split


# In[428]:


X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=48)


# In[429]:


from sklearn.linear_model import LogisticRegression


# In[430]:


Logistic = LogisticRegression()


# In[431]:


Logistic.fit(X_train, y_train)


# In[432]:


y_predict= Logistic.predict(X_test)


# In[433]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# In[434]:


print(accuracy_score(y_test,y_predict))


# In[435]:


print(confusion_matrix(y_test,y_predict))


# In[436]:


print(classification_report(y_test,y_predict))


# In[ ]:





# In[ ]:





# In[437]:


#Grid search hyper paramters


# In[438]:


model=LogisticRegression()


# In[439]:


penalty = ['l1', 'l2', 'elasticnet', 'none']
c_values = [100, 10, 1.0, 0.1, 0.01]
solver = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
C = [0.1, 1, 10]
max_iter= [100, 200, 300]


# In[440]:


param =dict(penalty=penalty, C=c_values,solver=solver,max_iter=max_iter)


# In[441]:


from sklearn.model_selection import GridSearchCV, StratifiedKFold


# In[442]:


cv = StratifiedKFold(n_splits=5)


# In[443]:


grid = GridSearchCV(estimator=model, param_grid=param, scoring="accuracy", cv=cv, n_jobs=1)


# In[444]:


grid.fit(X_train, y_train)


# In[445]:


y_pred=grid.predict(X_test)


# In[446]:


print(accuracy_score(y_test,y_pred))


# In[447]:


print(classification_report(y_test,y_predict))


# In[ ]:





# In[ ]:





# In[448]:


#Randomized Searc cv


# In[449]:


from sklearn.model_selection import RandomizedSearchCV


# In[450]:


model=LogisticRegression()


# In[451]:


randomcv=RandomizedSearchCV(estimator=model, param_distributions=param, cv=5, scoring="accuracy", random_state=48)


# In[452]:


randomcv.fit(X_train,y_train)


# In[453]:


y_predict=randomcv.predict(X_test)


# In[454]:


print(accuracy_score(y_test, y_predict))


# In[455]:


print(confusion_matrix(y_test,y_predict))


# In[456]:


print(classification_report(y_test,y_predict))

