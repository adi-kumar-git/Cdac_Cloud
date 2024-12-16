#!/usr/bin/env python
# coding: utf-8

# In[427]:


import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import warnings


# In[428]:


sns.load_dataset("tips")


# In[429]:


warnings.filterwarnings("ignore")


# In[430]:


df = sns.load_dataset("tips")


# In[431]:


X = df[['total_bill', 'sex', 'smoker', 'day', 'time', 'size']]


# In[432]:


y = df['tip'] > df['total_bill'] * 0.15


# In[433]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[434]:


from sklearn.preprocessing import LabelEncoder


# In[435]:


lb1= LabelEncoder()


# In[436]:


X_train['sex'] = lb1.fit_transform(X_train['sex'])
X_test['sex'] = lb1.transform(X_test['sex'])


# In[437]:


lb2= LabelEncoder()


# In[438]:


X_train['smoker'] = lb2.fit_transform(X_train['smoker'])
X_test['smoker'] = lb2.transform(X_test['smoker'])


# In[439]:


lb3= LabelEncoder()


# In[440]:


X_train['time'] = lb3.fit_transform(X_train['time'])
X_test['time'] = lb3.transform(X_test['time'])


# In[441]:


onehot_encoder = OneHotEncoder(sparse=False)


# In[442]:


X_train_day_encoded= onehot_encoder.fit_transform(X_train[["day"]])
X_test_day_encoded=onehot_encoder.transform(X_test[["day"]])


# In[443]:


X_train_day_encoded = pd.DataFrame(X_train_day_encoded, columns=onehot_encoder.get_feature_names_out(['day']))
X_test_day_encoded = pd.DataFrame(X_test_day_encoded, columns=onehot_encoder.get_feature_names_out(['day']))


# In[444]:


X_train=X_train.drop("day", axis=1).reset_index(drop=True)
X_test=X_test.drop("day", axis=1).reset_index(drop=True)


# In[445]:


X_train = pd.concat([X_train, X_train_day_encoded], axis=1)
X_test = pd.concat([X_test, X_test_day_encoded], axis=1)


# In[448]:


from sklearn.naive_bayes import GaussianNB


# In[449]:


nb_model=GaussianNB()


# In[450]:


nb_model.fit(X_train, y_train)


# In[451]:


y_predict=nb_model.predict(X_test)


# In[455]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[457]:


accuracy = accuracy_score(y_test, y_predict)
conf_matrix = confusion_matrix(y_test, y_predict)
classification_report = classification_report(y_test, y_predict)


# In[458]:


print(accuracy)


# In[459]:


print(conf_matrix)


# In[460]:


print(classification_report)

