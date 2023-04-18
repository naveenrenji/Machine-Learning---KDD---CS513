#!/usr/bin/env python
# coding: utf-8

# # Name : Naveen Mathews Renji | CWID : 20016323 

# ## Import Libraries

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# ## Load the dataset

# In[2]:


df = pd.read_csv("breast-cancer-wisconsin.csv")


# In[3]:


df.head()


# In[4]:


num_rows = df.shape[0]
print(num_rows)


# ## Taking care of missing values

# In[5]:


df = df.replace('?', float('nan'))
df["F6"] = pd.to_numeric(df["F6"])
df.describe()


# ## Replacing the missing values with the “mean” of the column

# In[6]:


mean_value=df['F6'].mean()
df['F6'].fillna(value=mean_value, inplace=True)
print(df.isnull().sum())


# ## Creating KNN Model and classifying the training data

# In[7]:


X = np.array(df.iloc[:, 1:])
y = np.array(df['Class'])


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.30, random_state = 42)


# In[9]:


neighbours_size=[3,5,10]
for i in neighbours_size:
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    Y_pred = knn.predict(X_test)
    print("The Classifiction models for KNN methodology for k Value = ",i)
    print("The Score is -",knn.score(X_test, y_test))
    print("The Confusion Matrix is -")
    print(confusion_matrix(y_test,Y_pred))
    print("The Classification Report Matrix is -")
    print(classification_report(y_test,Y_pred))

