#!/usr/bin/env python
# coding: utf-8

# # CS 513 B – KDD PROJECT 
# 
# ## Project Group No: 1
# 
# ## Forecasting Cancellation Flags: A Data-driven Approach to Hotel Reservation Cancellation Prediction
# 
# Problem Statement:
# The high rate of cancella/ons and no-shows in online hotel reserva/ons has become a challenge for hotels as it impacts their revenue and occupancy rates. While customers benefit from the flexibility of free or low-cost cancella/ons, hotels have to deal with the revenue-diminishing effect of empty rooms. Hence, there is a need to explore strategies that can help hotels reduce cancella/ons and no-shows while maintaining customer sa/sfac/on and loyalty. Therefore, the problem at hand is to predict the cancella-on flag of a hotel booking based on a set of features, in order to assist hotels in managing their resources and revenue more efficiently.
# 
# Source of Dataset: https://www.kaggle.com/datasets/naveenrenji/hotel-resource-management-dataset
# 
# Team Members: Group 1
# 1. Naveen Mathews Renji
# 2. Aatish Kayyath
# 3. Madhura Shinde
# 4. Abhishek Kocharekar

# ## Importing Libraries

# In[22]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from scipy import stats
import os
import sys
import time

# Visuals
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pandas.plotting import scatter_matrix
from sklearn.tree import plot_tree

# Scaling Solutions
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Automated Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

# models
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# Evaluation Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix


# ## Loading the dataset

# In[2]:


dataframe_one = pd.read_csv("Hotel-Reservations.csv")


# In[3]:


dataframe_one.head()


# In[4]:


# Print the ammount of rows and columns in the dataframe
print("[SHAPE BREAKDOWN]\n{} rows and {} columns".format(dataframe_one.shape[0], dataframe_one.shape[1]))

# Show the names of each column in the dataframe
print("\n[COLUMN BREAKDOWN]")
print(dataframe_one.columns)

# Print the first 10 rows of the dataframe
print("\n[FIRST 10 ROWS PREVIEW]")
dataframe_one.head(10)


# In[5]:


# Show the number of unique values in each column
print("[UNIQUE VALUES PER COLUMN]\n")
dataframe_one.nunique()


# ## Pre-processing the Data

# In[6]:


# SHow the number of numerical and non-numerical columns in the dataframe
print("[PRE CONVERSION]\n Number of numerical columns: {}".format(dataframe_one.select_dtypes(include=np.number).shape[1]))
print("\n[PRE CONVERSION]\n Number of non-numerical columns: {}".format(dataframe_one.select_dtypes(exclude=np.number).shape[1]))


# In[7]:


# SHow cleaning steps taken message
print("\n[CONVERSIONS MADE]")

# Drop the "Booking_ID" column
dataframe_one.drop("Booking_ID", axis=1, inplace=True)
print("- Dropped 'Booking_ID' column")

# Replace values that say "Not_Canceled" in the "booking_status" column with 0, and values that say "Canceled" with 1
dataframe_one["booking_status"].replace("Not_Canceled", 0, inplace=True)
dataframe_one["booking_status"].replace("Canceled", 1, inplace=True)
print("- Replaced values that say 'Not_Canceled' in the 'booking_status' column with 0, and values that say 'Canceled' with 1")

# Replace values that say "Not selected" in the "type_of_meal_plan" column with 0
dataframe_one["type_of_meal_plan"].replace("Not Selected", 0, inplace=True)
print("- Replaced values that say 'Not selected' in the 'type_of_meal_plan' column with 0")

# Replace values that say "Meal Plan 1" in the "type_of_meal_plan" column with 1
dataframe_one["type_of_meal_plan"].replace("Meal Plan 1", 1, inplace=True)
print("- Replaced values that say 'Meal Plan 1' in the 'type_of_meal_plan' column with 1")

# Replace values that say "Meal Plan 2" in the "type_of_meal_plan" column with 2
dataframe_one["type_of_meal_plan"].replace("Meal Plan 2", 2, inplace=True)
print("- Replaced values that say 'Meal Plan 2' in the 'type_of_meal_plan' column with 2")

# Replace values that say "Meal Plan 3" in the "type_of_meal_plan" column with 3
dataframe_one["type_of_meal_plan"].replace("Meal Plan 3", 3, inplace=True)
print("- Replaced values that say 'Meal Plan 3' in the 'type_of_meal_plan' column with 3")

# Replace values that say "Room_Type 1" in the "room_type_reserved" column with 1
dataframe_one["room_type_reserved"].replace("Room_Type 1", 1, inplace=True)
print("- Replaced values that say 'Room_Type 1' in the 'room_type_reserved' column with 1")

# Replace values that say "Room_Type 2" in the "room_type_reserved" column with 2
dataframe_one["room_type_reserved"].replace("Room_Type 2", 2, inplace=True)
print("- Replaced values that say 'Room_Type 2' in the 'room_type_reserved' column with 2")

# Replace values that say "Room_Type 3" in the "room_type_reserved" column with 3
dataframe_one["room_type_reserved"].replace("Room_Type 3", 3, inplace=True)
print("- Replaced values that say 'Room_Type 3' in the 'room_type_reserved' column with 3")

# Replace values that say "Room_Type 4" in the "room_type_reserved" column with 4
dataframe_one["room_type_reserved"].replace("Room_Type 4", 4, inplace=True)
print("- Replaced values that say 'Room_Type 4' in the 'room_type_reserved' column with 4")

# Replace values that say "Room_Type 5" in the "room_type_reserved" column with 5
dataframe_one["room_type_reserved"].replace("Room_Type 5", 5, inplace=True)
print("- Replaced values that say 'Room_Type 5' in the 'room_type_reserved' column with 5")

# Replace values that say "Room_Type 6" in the "room_type_reserved" column with 6
dataframe_one["room_type_reserved"].replace("Room_Type 6", 6, inplace=True)
print("- Replaced values that say 'Room_Type 6' in the 'room_type_reserved' column with 6")

# Replace values that say "Room_Type 7" in the "room_type_reserved" column with 7
dataframe_one["room_type_reserved"].replace("Room_Type 7", 7, inplace=True)
print("- Replaced values that say 'Room_Type 7' in the 'room_type_reserved' column with 7")

# Convert the unique text values in the "market_segment_type" column to numeric values
dataframe_one["market_segment_type"] = dataframe_one["market_segment_type"].map({"Offline": 0, "Online": 1, "Corporate": 2, "Aviation": 3, "Complementary": 4})
print("- Converted the unique text values in the 'market_segment_type' column to numeric values")

# Show number of numerical and non-numerical columns in the dataframe
print("\n[POST CONVERSION]\n Number of numerical columns: {}".format(dataframe_one.select_dtypes(include=np.number).shape[1]))
print("\n[POST CONVERSION]\n Number of non-numerical columns: {}".format(dataframe_one.select_dtypes(exclude=np.number).shape[1]))


# In[ ]:


# Convert all values in the dataframe to numeric
dataframe_one = dataframe_one.apply(pd.to_numeric, errors='coerce')


# In[8]:


# Print the first 10 rows of the dataframe
print("\n[FIRST 10 ROWS PREVIEW]")
dataframe_one.head(10)


# ## Checking for missing values

# In[9]:


print("[PRE FILLING]\n Total missing values is {}".format(dataframe_one.isnull().sum().sum()))
print("\n[PRE FILLING]\n Missing values by column is as follows:")
dataframe_one.isnull().sum()


# ## EDA - Pearson Correlation Heatmap

# In[11]:


plt.figure(figsize=(10,10))
sns.heatmap(dataframe_one.corr(method='pearson'), annot=True, fmt='.0%')
plt.show()


# ## EDA - FREQUENCY DISTRIBUTION

# In[12]:


# Show the distribution of values in each column of the dataframe
dataframe_one.hist(figsize=(20, 20))
plt.show()


# ## EDA - VIOLIN PLOTS

# In[13]:


# Show violin plots of each column in the dataframe
plt.figure(figsize=(20, 60))
for i, column in enumerate(dataframe_one.columns):
    plt.subplot(12, 3, i+1)
    sns.violinplot(dataframe_one[column])
    plt.title(column)
plt.show()


# ## Taking care of outliers

# In[14]:


# Preview the effects of multiple outlier removal methods on the dataframe and show how many outliers each would remove
print("[OUTLIER REMOVAL METHOD]\n Z=SCORE BASED OUTLIER REMOVAL\n")

# Set the name of the target column
target_column = 'booking_status'                                             

# Create a copy of the dataframe
dataframe_two = dataframe_one.copy()

# Separate the target column from the feature columns
target = dataframe_two[target_column]
features = dataframe_two.drop(target_column, axis=1)

# Create a copy of the dataframe
dataframe_final = dataframe_one.copy()

# Separate the target column from the feature columns
target = dataframe_final[target_column]
features = dataframe_final.drop(target_column, axis=1)

# Remove outliers using the Z-Score method
z_scores = np.abs(stats.zscore(features))
features = features[(z_scores < 3).all(axis=1)]

# Get the cleaned feature indices
cleaned_feature_indices = features.index

# Combine the target column with the cleaned feature columns
dataframe_final = pd.concat([target.iloc[cleaned_feature_indices], features], axis=1)
dataframe_final.reset_index(drop=True, inplace=True)

# Print the number of rows in the dataframe before and after Z-Score outlier removal
print("PRE Z-SCORE OUTLIER REMOVAL ROWS: {}".format(dataframe_one.shape[0]))
print("POST Z-SCORE OUTLIER REMOVAL ROWS: {}".format(dataframe_final.shape[0]))


# ## Splitting the Data for Training and Testing

# In[15]:


# SEE TOTAL SAMPLES FOR EACH CLASS
class_count = dataframe_final["booking_status"].value_counts()
print("\n[SEE TOTAL SAMPLES FOR EACH CLASS]")
print(class_count)


# In[17]:


# Split the data into training and testing sets using the "Loan_Status" column as the target
X = dataframe_final.drop('booking_status', axis=1)
y = dataframe_final['booking_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Show the shapes of the training and testing sets for both the features and the target
print("\n[TRAINING SET FEATURES SHAPE]\n {}".format(X_train.shape))
print("\n[TRAINING SET TARGET SHAPE]\n {}".format(y_train.shape))
print("\n[TESTING SET FEATURES SHAPE]\n {}".format(X_test.shape))
print("\n[TESTING SET TARGET SHAPE]\n {}".format(y_test.shape))


# ## SCALE DATA

# In[20]:


# Scale the training and testing data using ZScoreScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("[DATA SCALED VIA Z-SCORE]")


# ## Random Forest Model

# In[23]:


# Create an random forest classifier model
rf = RandomForestClassifier(random_state=42)

# Create a dictionary of all the parameters you want to tune
params_rf = {'n_estimators': [100, 200, 300],
             'max_depth': [2, 4, 6, 8, 10],
             'min_samples_split': [2, 5, 10, 15],
             'min_samples_leaf': [1, 2, 5, 10]}

# Use grid search to test all the possible combinations of parameters
grid = GridSearchCV(rf, params_rf, cv=5, verbose=1, n_jobs=-1)

# Fit the model to the training data and print the best parameters
grid.fit(X_train, y_train)

# Make a new random forest classifier model with the best parameters
rf = RandomForestClassifier(n_estimators=grid.best_params_['n_estimators'],
                            max_depth=grid.best_params_['max_depth'],
                            min_samples_split=grid.best_params_['min_samples_split'],
                            min_samples_leaf=grid.best_params_['min_samples_leaf'],
                            random_state=42)

# Train the model on the training data
rf.fit(X_train, y_train)


# Make predictions on the test data
y_pred = rf.predict(X_test)


# ## Results and Evaluation

# In[24]:


# Print the best parameters found by the grid search
print("BEST PARAMETERS:\n {}".format(grid.best_params_))

# Print the accuracy, precision, recall, and F1 score of the classifier for both the training and testing data
print("\nTRAINING ACCURACY: {:.2%}".format(accuracy_score(y_train, rf.predict(X_train))))
print("TESTING ACCURACY: {:.2%}".format(accuracy_score(y_test, y_pred)))
print("\nTRAINING PRECISION: {:.2%}".format(precision_score(y_train, rf.predict(X_train))))
print("TESTING PRECISION: {:.2%}".format(precision_score(y_test, y_pred)))
print("\nTRAINING RECALL: {:.2%}".format(recall_score(y_train, rf.predict(X_train))))
print("TESTING RECALL: {:.2%}".format(recall_score(y_test, y_pred)))
print("\nTRAINING F1 SCORE: {:.2%}".format(f1_score(y_train, rf.predict(X_train))))
print("TESTING F1 SCORE: {:.2%}".format(f1_score(y_test, y_pred)))

# Print a confusion matrix for the testing data
print("\nCONFUSION MATRIX:\n {}".format(confusion_matrix(y_test, y_pred)))


# ## Prediction Plots

# In[27]:


# Plot the the correct and incorrect predictions of the classifier
sns.set_style('whitegrid')
sns.set_context('poster')
sns.set_palette('colorblind')
sns.set(rc={'figure.figsize':(12,8)})
sns.barplot(x=['CORRECT', 'INCORRECT'], y=[accuracy_score(y_test, y_pred), 1-accuracy_score(y_test, y_pred)])
plt.title('CORRECT VS INCORRECT PREDICTIONS (TOTALS)') 
plt.show()


# ## Feature importance plot

# In[28]:


# Plot the feature importance of the classifier using seaborn
sns.set_style('whitegrid')
sns.set_context('poster')
sns.set_palette('colorblind')
sns.set(rc={'figure.figsize':(12,8)})
sns.barplot(x=rf.feature_importances_, y=X.columns)
plt.title('FEATURE IMPORTANCE')
plt.show()


# ## KNN Model 

# In[29]:


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


# In[ ]:




