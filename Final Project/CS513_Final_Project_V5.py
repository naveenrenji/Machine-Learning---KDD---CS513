#!/usr/bin/env python
# coding: utf-8

# 
# #### chagelog v5
# 1. Added ANN
# 
# #### chagelog v4
# 1. Added logistic regression, decision tree, LightGBM, CatBoost, XGBoost, SVM
# 
# #### changelogs v3
# 1. Added SMOTE upscaling to balance data - increased accuracy by 2% on average.(GBM is a 90% accuracy now)
# 
# 
# 
# #### changelogs v2
# 1. Added ADABOOST and GradientBoost
# 2. Added more EDA - added distribution of every variable
# 3. OHE added for the categorical variables

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

# In[2]:


import warnings
warnings.filterwarnings('ignore')

# Importing necessary libraries
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from scipy import stats
import os
import sys
import time

# Importing libraries for visuals
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pandas.plotting import scatter_matrix
from sklearn.tree import plot_tree

# Importing libraries for scaling
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

# In[3]:


from google.colab import files
uploaded = files.upload()


# In[4]:


df = pd.read_csv("Hotel-Reservations.csv")


# In[5]:


df.head()


# In[6]:


# Print the ammount of rows and columns in the dataframe
rows, cols = df.shape
print(f"SHAPE BREAKDOWN\n{rows} rows and {cols} columns")


# Show the names of each column in the dataframe
print("\nCOLUMN BREAKDOWN")
print(df.columns)

print("\nFIRST 15 ROWS")
df.head(15)


# In[7]:


print("UNIQUE VALUES IN EACH COLUMN\n")
df.nunique()


# ### Number of Visitors

# In[8]:


df.groupby('no_of_adults')['Booking_ID'].agg(['count']).sort_values(by='count',ascending=False).plot(kind='pie',autopct='%1.2f%%',subplots=True,title='Adults',figsize=(9,9))


# In[9]:


df.groupby('no_of_children')['Booking_ID'].agg(['count']).sort_values(by='count',ascending=False).plot(kind='pie',autopct='%1.2f%%',subplots=True,title='Children',figsize=(9,9))


# ### Nights

# In[10]:


df.groupby('no_of_weekend_nights')['Booking_ID'].agg(['count']).sort_values(by='count',ascending=False).plot(kind='bar',figsize=(9,9))


# In[12]:


df.groupby('no_of_week_nights')['Booking_ID'].agg(['count']).sort_values(by='count',ascending=False).plot(kind='barh',figsize=(9,9))


# In[13]:


sns.histplot(x='no_of_week_nights',data=df,hue='no_of_weekend_nights',kde=True,palette='Set3');


# ### Meal plans

# In[14]:


df.groupby('type_of_meal_plan')['Booking_ID'].agg(['count']).sort_values(by='count',ascending=False).plot(kind='pie',autopct='%1.2f%%',subplots=True,title='Meal',figsize=(9,9))


# ### Parking spaces

# In[15]:


df.groupby('required_car_parking_space')['Booking_ID'].agg(['count']).sort_values(by='count',ascending=False).plot(kind='pie',autopct='%1.2f%%',subplots=True,title='Required car parking space',figsize=(9,9))


# ### Room Type

# In[16]:


df.groupby('room_type_reserved')['Booking_ID'].agg(['count']).sort_values(by='count',ascending=False).plot(kind='barh',figsize=(9,9))


# ### Distribution by years, months, seasons, days

# In[17]:


df.groupby('arrival_year')['Booking_ID'].agg(['count']).sort_values(by='count',ascending=False).plot(kind='pie',autopct='%1.2f%%',subplots=True,title='Years',figsize=(9,9))


# In[18]:


df.groupby('arrival_month')['Booking_ID'].agg(['count']).sort_values(by='count',ascending=False).plot(kind='pie',autopct='%1.2f%%',subplots=True,title='Months',figsize=(9,9))


# In[11]:


def season(x):
    if x in [9,10,11]:
        return 'Autumn'
    if x in [1,2,12]:
        return 'Winter'
    if x in [3,4,5]:
        return 'Spring'
    if x in [6,7,8]:
        return 'Summer'
    return x


# In[12]:


df['season_group']=df['arrival_month'].apply(season)


# In[13]:


df.groupby('season_group')['Booking_ID'].agg(['count']).sort_values(by='count',ascending=False).plot(kind='pie',autopct='%1.2f%%',subplots=True,title='Seasons',figsize=(9,9))


# In[14]:


df.groupby('arrival_date')['Booking_ID'].agg(['count']).sort_values(by='count',ascending=False).plot(kind='pie',autopct='%1.2f%%',subplots=True,title='Date',figsize=(9,9))


# In[15]:


df.pivot_table(index='arrival_year',columns='arrival_month',values='arrival_date', aggfunc=(['count']))


# ### Segments

# In[16]:


df.groupby('market_segment_type')['Booking_ID'].agg(['count']).sort_values(by='count',ascending=False).plot(kind='pie',autopct='%1.2f%%',subplots=True,title='Types',figsize=(9,9))


# ### Repeated guest

# In[17]:


df.groupby('repeated_guest')['Booking_ID'].agg(['count']).sort_values(by='count',ascending=False).plot(kind='pie',autopct='%1.2f%%',subplots=True,title='Repeated guest',figsize=(9,9))


# ### Cancellations in the past

# In[18]:


df.groupby('no_of_previous_cancellations')['Booking_ID'].agg(['count']).sort_values(by='count',ascending=False).plot(kind='pie',autopct='%1.3f%%',subplots=True,title='Cancellations',figsize=(9,9))


# ### 0 cancellations

# In[19]:


df.groupby('no_of_previous_bookings_not_canceled')['Booking_ID'].agg(['count']).sort_values(by='count',ascending=False).plot(kind='bar',title='0 cancellations',figsize=(9,9))


# ### Average Price

# In[20]:


def avg_price_per_room_group(x):
    if x <= 50.0 :
        x= 'price below 50'
    elif x >50.0 and x <=150.0:
        x= 'price from 50 to 150'
    elif x >150.0 and x <=300.0:
        x= 'price from 150 to 300'
    elif x >300.0 and x <=450.0:
        x= 'price from 300 to 450'
    else:
        x= 'price 450+'
    return x


# In[21]:


df['price_per_room_group']=df['avg_price_per_room'].apply(avg_price_per_room_group)


# In[22]:


df.groupby('price_per_room_group')['Booking_ID'].agg(['count']).sort_values(by='count',ascending=False).plot(kind='barh',figsize=(9,9))


# ### Special Requests

# In[23]:


df.groupby('no_of_special_requests')['Booking_ID'].agg(['count']).sort_values(by='count',ascending=False).plot(kind='bar',figsize=(9,9))


# ### Booking Status

# In[24]:


df.groupby('booking_status')['Booking_ID'].agg(['count']).sort_values(by='count',ascending=False).plot(kind='pie',autopct='%1.2f%%',subplots=True,title='Status',figsize=(9,9))


# In[25]:


# Drop the "Booking_ID" column
df.drop("Booking_ID", axis=1, inplace=True)
print("- Dropped 'Booking_ID' column")


# ## EDA - Pearson Correlation Heatmap
# 

# In[26]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(method='pearson'), annot=True, fmt='.0%')
plt.show()


# ## EDA - FREQUENCY DISTRIBUTION

# In[27]:


df.hist(figsize=(20, 20))
plt.show()


# ## Pre-processing the Data

# In[28]:


# Finding the number of numerical and non-numerical columns
print("PRE CONVERSION\n Number of numerical columns: {}".format(df.select_dtypes(include=np.number).shape[1]))
print("\nPRE CONVERSION\n Number of non-numerical columns: {}".format(df.select_dtypes(exclude=np.number).shape[1]))


# In[29]:


df.head()


# In[30]:


# Define a dictionary of replacements to be made in the dataframe
replacements = {
    "booking_status": {"Not_Canceled": 0, "Canceled": 1},
    }

# Replace values in the dataframe using the dictionary
df.replace(replacements, inplace=True)


# In[31]:


non_numeric_columns = list(df.select_dtypes(exclude=['number']).columns)


# In[32]:


non_numeric_columns


# In[33]:



# Apply one-hot encoding to each non-numerical column
for column in non_numeric_columns:
    # Generate the one-hot encoded variables for the current column
    dummies = pd.get_dummies(df[column], prefix=column)
    # Add the one-hot encoded variables to the original dataframe
    df = pd.concat([df, dummies], axis=1)
    # Remove the original column from the dataframe
    df.drop(column, axis=1, inplace=True)


# In[34]:


# Show number of numerical and non-numerical columns in the dataframe
print("\nPOST CONVERSION\n Number of numerical columns: {}".format(df.select_dtypes(include=np.number).shape[1]))
print("\nPOST CONVERSION\n Number of non-numerical columns: {}".format(df.select_dtypes(exclude=np.number).shape[1]))


# In[35]:


print("\nFIRST 15 ROWS")
df.head(15)


# ## Checking for missing values

# In[36]:


print("Total missing values is {}".format(df.isnull().sum().sum()))
print("\nMissing values by column is as follows:")
df.isnull().sum()


# ## Taking care of outliers

# In[37]:


print("Z=SCORE BASED OUTLIER REMOVAL\n")

target_column = 'booking_status'                                             

dataframe_final = df.copy()

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

print(f"PRE Z-SCORE OUTLIER REMOVAL - ROWS: {df.shape[0]}")
print(f"POST Z-SCORE OUTLIER REMOVAL - ROWS: {dataframe_final.shape[0]}")


# In[38]:


dataframe_final


# ## Splitting the Data for Training and Testing

# In[39]:


# SEE TOTAL SAMPLES FOR EACH CLASS
class_count = dataframe_final["booking_status"].value_counts()
print("\nTOTAL SAMPLES FOR EACH CLASS")
print(class_count)


# In[40]:


pip install imblearn


# In[41]:


from imblearn.over_sampling import SMOTE


# In[42]:


X = dataframe_final.drop('booking_status', axis=1)
y = dataframe_final['booking_status']


# In[43]:


# Apply SMOTE to oversample minority class
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Create new balanced DataFrame
balanced_df = pd.concat([X_res, y_res], axis=1)


# In[44]:


# SEE TOTAL SAMPLES FOR EACH CLASS
class_count = balanced_df["booking_status"].value_counts()
print("\nTOTAL SAMPLES FOR EACH CLASS")
print(class_count)


# In[45]:


X = balanced_df.drop('booking_status', axis=1)
y = balanced_df['booking_status']


# In[46]:



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# ## SCALE DATA

# In[47]:


# Scale the training and testing data using ZScoreScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("DATA SCALED VIA Z-SCORE")


# # USING DIFFERENT MACHINE LEARNING ALGORITHMS

# ## 1. Random Forest Model

# In[ ]:


rf = RandomForestClassifier(random_state=42)

# Define a dictionary of all the parameters to tune
params_rf = {'n_estimators': [100, 200, 300],
             'max_depth': [2, 4, 6, 8, 10],
             'min_samples_split': [2, 5, 10, 15],
             'min_samples_leaf': [1, 2, 5, 10]}

# Use grid search to test all possible combinations of parameters
grid = GridSearchCV(rf, params_rf, cv=5, verbose=1, n_jobs=-1)

# Fit the model to the training data and print the best parameters
grid.fit(X_train, y_train)
print(f"Best parameters: {grid.best_params_}")

# Update the existing model with the best parameters
rf.set_params(**grid.best_params_)

# Train the model on the training data
rf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf.predict(X_test)


# ## Results and Evaluation

# In[ ]:


best_params = grid.best_params_
train_accuracy = accuracy_score(y_train, rf.predict(X_train))
test_accuracy = accuracy_score(y_test, y_pred)
train_precision = precision_score(y_train, rf.predict(X_train))
test_precision = precision_score(y_test, y_pred)
train_recall = recall_score(y_train, rf.predict(X_train))
test_recall = recall_score(y_test, y_pred)
train_f1_score = f1_score(y_train, rf.predict(X_train))
test_f1_score = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"BEST PARAMETERS:\n {best_params}\n")
print(f"TRAINING ACCURACY: {train_accuracy:.2%}")
print(f"TESTING ACCURACY: {test_accuracy:.2%}\n")
print(f"TRAINING PRECISION: {train_precision:.2%}")
print(f"TESTING PRECISION: {test_precision:.2%}\n")
print(f"TRAINING RECALL: {train_recall:.2%}")
print(f"TESTING RECALL: {test_recall:.2%}\n")
print(f"TRAINING F1 SCORE: {train_f1_score:.2%}")
print(f"TESTING F1 SCORE: {test_f1_score:.2%}\n")
print(f"CONFUSION MATRIX:\n {conf_matrix}")


# ## Prediction Plots

# In[ ]:


num_correct = accuracy_score(y_test, y_pred)
num_incorrect = 1 - num_correct

sns.set(style='whitegrid', context='poster', palette='colorblind', rc={'figure.figsize': (12, 8)})
sns.barplot(x=['CORRECT', 'INCORRECT'], y=[num_correct, num_incorrect])
plt.title('RANDOM FOREST CORRECT VS INCORRECT PREDICTIONS')
plt.show()


# ## Feature importance plot

# In[ ]:


# Create a figure and axis objects using matplotlib
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the feature importance of the classifier using seaborn
sns.set_style('whitegrid')
sns.set_context('poster')
sns.set_palette('colorblind')
sns.barplot(x=rf.feature_importances_, y=X.columns, ax=ax)
ax.set_title('FEATURE IMPORTANCE')

# Display the plot
plt.show()


# # 2. KNN Model 

# In[48]:


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


# ## Prediction Plot

# In[49]:


num_correct = accuracy_score(y_test, Y_pred)
num_incorrect = 1 - num_correct

sns.set(style='whitegrid', context='poster', palette='colorblind', rc={'figure.figsize': (12, 8)})
sns.barplot(x=['CORRECT', 'INCORRECT'], y=[num_correct, num_incorrect])
plt.title('KNN CORRECT VS INCORRECT PREDICTIONS')
plt.show()


# ## GRADIENT BOOSTING

# In[50]:


from sklearn.ensemble import GradientBoostingClassifier

n_estimators = np.array([300])
learning_rate = np.array([0.01, 0.02, 0.05, 0.07, 0.09, 0.1, 0.3, 0.5, 0.005])
values_grid = {'n_estimators': n_estimators, 'learning_rate': learning_rate}

model = GradientBoostingClassifier()
gridGradient = GridSearchCV(estimator = model, param_grid = values_grid, cv = 5, n_jobs = -1)
gridGradient.fit(X_train, y_train)


# In[51]:


print('Learning Rate: ', gridGradient.best_estimator_.learning_rate)
print('Score: ', gridGradient.best_score_)


# In[52]:


grad_boost = GradientBoostingClassifier(n_estimators = 300, learning_rate = 0.5, random_state = 0)
grad_boost.fit(X_train, y_train)
previsoes = grad_boost.predict(X_test)


# In[53]:


print("The Confusion Matrix is -")
print(confusion_matrix(y_test,previsoes))
print("The Classification Report Matrix is -")
print(classification_report(y_test,previsoes))


# ## Prediction Plots

# In[54]:


num_correct = accuracy_score(y_test, previsoes)
num_incorrect = 1 - num_correct

sns.set(style='whitegrid', context='poster', palette='colorblind', rc={'figure.figsize': (12, 8)})
sns.barplot(x=['CORRECT', 'INCORRECT'], y=[num_correct, num_incorrect])
plt.title('GRADIENT BOOST CORRECT VS INCORRECT PREDICTIONS')
plt.show()


# ## ADABOOST

# In[55]:


from sklearn.ensemble import AdaBoostClassifier

n_estimators = np.array([500])
learning_rate = np.array([2.0, 2.5, 1.9, 1.7, 0.5, 0.4])
values_grid = {'n_estimators': n_estimators, 'learning_rate': learning_rate}

model = AdaBoostClassifier()
gridAdaBoost = GridSearchCV(estimator = model, param_grid = values_grid, cv = 5, n_jobs = -1)
gridAdaBoost.fit(X_train, y_train)


# In[56]:


print('Learning Rate: ', gridAdaBoost.best_estimator_.learning_rate)
print('Score: ', gridAdaBoost.best_score_)


# In[57]:


ada_boost = AdaBoostClassifier(n_estimators = 500, learning_rate = 1.9, random_state = 0)
ada_boost.fit(X_train, y_train)
y_pred_AB = ada_boost.predict(X_test)


# In[58]:


print("The Confusion Matrix is -")
print(confusion_matrix(y_test,y_pred_AB))
print("The Classification Report Matrix is -")
print(classification_report(y_test,y_pred_AB))


# ### PREDICTION PLOTS

# In[59]:


num_correct = accuracy_score(y_test, y_pred_AB)
num_incorrect = 1 - num_correct

sns.set(style='whitegrid', context='poster', palette='colorblind', rc={'figure.figsize': (12, 8)})
sns.barplot(x=['CORRECT', 'INCORRECT'], y=[num_correct, num_incorrect])
plt.title('ADA BOOST CORRECT VS INCORRECT PREDICTIONS')
plt.show()


# ## Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


log_model = LogisticRegression().fit(X_train, y_train)


# In[ ]:


y_pred_log = log_model.predict(X_test)


# In[ ]:


print("The Confusion Matrix is -")
print(confusion_matrix(y_test,y_pred_log))
print("The Classification Report Matrix is -")
print(classification_report(y_test,y_pred_log))


# ### PREDICTION PLOTS

# In[ ]:


num_correct = accuracy_score(y_test, y_pred_log)
num_incorrect = 1 - num_correct

sns.set(style='whitegrid', context='poster', palette='colorblind', rc={'figure.figsize': (12, 8)})
sns.barplot(x=['CORRECT', 'INCORRECT'], y=[num_correct, num_incorrect])
plt.title('LOGISTIC CORRECT VS INCORRECT PREDICTIONS')
plt.show()


# ## Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

def dtree_grid_search(X,y,nfolds):
    #create a dictionary of all values we want to test
    param_grid = { 'criterion':['gini','entropy'],'max_depth': np.arange(3, 15)}
    # decision tree model
    dtree_model=DecisionTreeClassifier()
    #use gridsearch to test all values
    dtree_gscv = GridSearchCV(dtree_model, param_grid, cv=nfolds)
    #fit model to data
    dtree_gscv.fit(X, y)
    return dtree_gscv.best_params_

dtree_grid_search(X_train, y_train, 5)


# In[ ]:


dtree = DecisionTreeClassifier(criterion='gini', max_depth=13).fit(X_train, y_train)


# In[ ]:


y_pred_tree = dtree.predict(X_test)


# In[ ]:


print("The Confusion Matrix is -")
print(confusion_matrix(y_test,y_pred_tree))
print("The Classification Report Matrix is -")
print(classification_report(y_test,y_pred_tree))


# ### PREDICTION PLOTS

# In[ ]:


num_correct = accuracy_score(y_test, y_pred_tree)
num_incorrect = 1 - num_correct

sns.set(style='whitegrid', context='poster', palette='colorblind', rc={'figure.figsize': (12, 8)})
sns.barplot(x=['CORRECT', 'INCORRECT'], y=[num_correct, num_incorrect])
plt.title('Decision Tree CORRECT VS INCORRECT PREDICTIONS')
plt.show()


# ## LightGBM

# In[ ]:


import lightgbm as ltb

lightgbm = ltb.LGBMClassifier().fit(X_train, y_train)


# In[ ]:


y_pred_light = lightgbm.predict(X_test)


# In[ ]:


print("The Confusion Matrix is -")
print(confusion_matrix(y_test,y_pred_light))
print("The Classification Report Matrix is -")
print(classification_report(y_test,y_pred_light))


# ### PREDICTION PLOTS

# In[ ]:


num_correct = accuracy_score(y_test, y_pred_light)
num_incorrect = 1 - num_correct

sns.set(style='whitegrid', context='poster', palette='colorblind', rc={'figure.figsize': (12, 8)})
sns.barplot(x=['CORRECT', 'INCORRECT'], y=[num_correct, num_incorrect])
plt.title('LIGHTGBM CORRECT VS INCORRECT PREDICTIONS')
plt.show()


# ## CatBoost

# In[ ]:


from catboost import CatBoostClassifier

cat = CatBoostClassifier().fit(X_train, y_train)


# In[ ]:


y_pred_cat = cat.predict(X_test)


# In[ ]:


print("The Confusion Matrix is -")
print(confusion_matrix(y_test,y_pred_cat))
print("The Classification Report Matrix is -")
print(classification_report(y_test,y_pred_cat))


# ### PREDICTION PLOTS

# In[ ]:


num_correct = accuracy_score(y_test, y_pred_cat)
num_incorrect = 1 - num_correct

sns.set(style='whitegrid', context='poster', palette='colorblind', rc={'figure.figsize': (12, 8)})
sns.barplot(x=['CORRECT', 'INCORRECT'], y=[num_correct, num_incorrect])
plt.title('CAT-BOOST CORRECT VS INCORRECT PREDICTIONS')
plt.show()


# ## XGBoost

# In[ ]:


import xgboost as xgb

xgboost = xgb.XGBClassifier().fit(X_train, y_train)


# In[ ]:


y_pred_xgb = xgboost.predict(X_test)


# In[ ]:


print("The Confusion Matrix is -")
print(confusion_matrix(y_test,y_pred_xgb))
print("The Classification Report Matrix is -")
print(classification_report(y_test,y_pred_xgb))


# ### PREDICTION PLOTS

# In[ ]:


num_correct = accuracy_score(y_test, y_pred_xgb)
num_incorrect = 1 - num_correct

sns.set(style='whitegrid', context='poster', palette='colorblind', rc={'figure.figsize': (12, 8)})
sns.barplot(x=['CORRECT', 'INCORRECT'], y=[num_correct, num_incorrect])
plt.title('XG BOOST CORRECT VS INCORRECT PREDICTIONS')
plt.show()


# ## SVM

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']} 
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
  
# fitting the model for grid search
grid.fit(X_train, y_train)


# In[ ]:


grid.best_params_


# In[ ]:


svm = SVC(C=10, gamma=0.1).fit(X_train, y_train)


# In[ ]:


y_pred_svm = svm.predict(X_test)


# In[ ]:


print("The Confusion Matrix is -")
print(confusion_matrix(y_test,y_pred_svm))
print("The Classification Report Matrix is -")
print(classification_report(y_test,y_pred_svm))

### PREDICTION PLOTS

num_correct = accuracy_score(y_test, y_pred_svm)
num_incorrect = 1 - num_correct

sns.set(style='whitegrid', context='poster', palette='colorblind', rc={'figure.figsize': (12, 8)})
sns.barplot(x=['CORRECT', 'INCORRECT'], y=[num_correct, num_incorrect])
plt.title('SVM CORRECT VS INCORRECT PREDICTIONS')
plt.show()


# In[ ]:





# ## ANN

# In[60]:


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# Set the seed for reproducibility
np.random.seed(42)

# Define the number of input features
input_dim = X_train.shape[1]

# Define the model architecture
model = Sequential()
model.add(Dense(128, input_dim=input_dim, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
adam = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

# Fit the model on the training data
model.fit(X_train, y_train, epochs=200, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')


# In[64]:


y_pred_con= model.predict(X_test)
y_pred_ann = np.round(y_pred_con)


# In[65]:


print("The Confusion Matrix is -")
print(confusion_matrix(y_test,y_pred_ann))
print("The Classification Report Matrix is -")
print(classification_report(y_test,y_pred_ann))

### PREDICTION PLOTS

num_correct = accuracy_score(y_test, y_pred_ann)
num_incorrect = 1 - num_correct

sns.set(style='whitegrid', context='poster', palette='colorblind', rc={'figure.figsize': (12, 8)})
sns.barplot(x=['CORRECT', 'INCORRECT'], y=[num_correct, num_incorrect])
plt.title('ANN CORRECT VS INCORRECT PREDICTIONS')
plt.show()


# In[ ]:




