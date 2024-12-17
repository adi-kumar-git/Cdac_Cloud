#!/usr/bin/env python
# coding: utf-8

# In[113]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[114]:


from sklearn.datasets import load_iris


# In[115]:


iris=load_iris()


# In[116]:


iris["DESCR"]


# In[117]:


print(iris["target"])


# In[118]:


X=pd.DataFrame(iris["data"], columns=["sepal_len","sepal_width","petal_len","petal_width"])


# In[119]:


X


# In[120]:


y=pd.DataFrame(iris["target"])


# In[121]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=48)


# In[122]:


#applying decision trees


# In[123]:


from sklearn.tree import DecisionTreeClassifier


# In[124]:


tree_classifier = DecisionTreeClassifier()


# In[125]:


tree_classifier.fit(X_train,y_train)


# In[126]:


y_predict=tree_classifier.predict(X_test)


# In[127]:


from sklearn.metrics import accuracy_score


# In[128]:


accuracy=accuracy_score(y_test,y_predict)


# In[129]:


print(f"Accuracy: {accuracy * 100:.2f}%")


# In[130]:


tree_classifier.fit(X_train, y_train)


# In[131]:


#vissulaize the decision tree


# In[132]:


from sklearn import tree


# In[133]:


plt.figure(figsize=(15,10))
tree.plot_tree(tree_classifier, filled= True)


# In[134]:


tree_post_prunning_classifier= DecisionTreeClassifier( criterion="entropy", max_depth=3)


# In[135]:


tree_post_prunning_classifier.fit(X_train,y_train)


# In[136]:


y_predict = tree_post_prunning_classifier.predict(X_test)


# In[137]:


acc_ppc=accuracy_score(y_test,y_predict)


# In[138]:


print(f"Accuracy: {acc_ppc * 100:.2f}%")


# In[139]:


#-----------------------------------------------------------------------------------------------


# In[140]:


param = {
    "criterion": ["gini", "entropy"],  # valid options
    "splitter": ["best", "random"],    # valid options
    "max_depth": [1, 2, 3, 4, 5],       # valid values for max_depth
    "max_features": ["auto", "sqrt", "log2"]  # valid options
}


# In[141]:


from sklearn.model_selection import GridSearchCV


# In[142]:


treemodel=DecisionTreeClassifier()


# In[143]:


grid=GridSearchCV(treemodel, param_grid=param, cv=5, scoring="accuracy")


# In[144]:


import warnings
warnings.filterwarnings("ignore")


# In[145]:


grid.fit(X_train, y_train)


# In[146]:


grid.best_params_


# In[147]:


grid.best_score_


# In[148]:


y_predict=grid.predict(X_test)


# In[149]:


acc_grid= accuracy_score(y_test,y_predict)


# In[151]:


print(f"Accuracy: {acc_ppc * 100:.2f}%")


# In[161]:


conf_matrix = confusion_matrix(y_test, y_predict)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Visualizing the confusion matrix using seaborn heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Classification Report
class_report = classification_report(y_test, y_predict, target_names=iris.target_names)

# Print the classification report
print("\nClassification Report:")
print(class_report)


# In[153]:


from sklearn.metrics import confusion_matrix, classification_report


# In[157]:


import seaborn as sns

