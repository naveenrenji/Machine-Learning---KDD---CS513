#!/usr/bin/env python
# coding: utf-8

# # Name : Naveen Mathews Renji | CWID : 20016323 

# ## Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# ## Load dataset from csv file

# In[2]:


df = pd.read_csv('breast-cancer-wisconsin.csv')
df.head()


# ## Drop the 'Sample' column

# In[3]:


del df['Sample']


# In[4]:


num_rows = df.shape[0]
print(num_rows)


# ## Replace missing values with mean and convert 'F6' column to float datatype

# In[5]:


df['F6'] = df['F6'].replace('?', np.nan)
df['F6'] = df['F6'].astype('float')
df['F6'].fillna(value = np.round(df['F6'].mean(), 4), inplace=True)


# ## Convert 'Class' column to categorical datatype

# In[6]:


df['Class'] = df['Class'].astype('category')


# ## Separate the target variable and feature variables

# In[7]:


y = df['Class']
X = df.drop('Class', axis=1).values


# ## Convert the target variable to binary classification problem

# In[8]:


y = np.where(y==2, 0, 1)


# ## Split the data into training and testing sets

# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# ## Train a decision tree classifier on the training set

# In[10]:


dTree = DecisionTreeClassifier(random_state=123)
dTree.fit(X_train, y_train)


# ## Evaluate the performance of the model on the testing set

# In[11]:


y_pred = dTree.predict(X_test)


# ## Results

# In[12]:


print(classification_report(y_test, y_pred))


# In[13]:


print('Accuracy:', accuracy_score(y_test, y_pred))

