# Clear environment variables
rm(list=ls())

# Import Libraries
library(tidyverse)
library(e1071)
library(caret)

# Load the dataset
df <- read.csv("breast-cancer-wisconsin.csv")

# Print the first few rows
head(df)

# Remove the first column as it is not needed
df <- df[,-1]

# Taking care of missing values
df <- df %>%
  mutate(F6 = ifelse(F6 == "?", NA, as.numeric(F6))) %>%
  drop_na()

# Split the data into training and test sets
set.seed(123) # For reproducibility
trainIndex <- createDataPartition(df$Class, p = .7, list = FALSE, times = 1)
train <- df[trainIndex, ]
test <- df[-trainIndex, ]

# Create NB Model and classify the training data
y_train <- ifelse(train$Class == 2, 0, 1)
y_test <- ifelse(test$Class == 2, 0, 1)
X_train <- train %>% select(-Class) %>% as.matrix()
X_test <- test %>% select(-Class) %>% as.matrix()

# Gaussian Naive Bayes
nbg <- naiveBayes(x = X_train, y = y_train, type = "raw")
Y_pred <- predict(nbg, X_test, type = "raw")
confusionMatrix(Y_pred, y_test, positive = "0")
confusionMatrix(Y_pred, y_test, positive = "0")$table

# Bernoulli Naive Bayes
nbb <- naiveBayes(x = X_train, y = y_train, type = "class")
Y_pred <- predict(nbb, X_test, type = "class")
confusionMatrix(Y_pred, y_test, positive = "0")
confusionMatrix(Y_pred, y_test, positive = "0")$table

# Multinomial Naive Bayes
nbm <- naiveBayes(x = X_train, y = y_train, type = "count")
Y_pred <- predict(nbm, X_test, type = "raw")
confusionMatrix(Y_pred, y_test, positive = "0")
confusionMatrix(Y_pred, y_test, positive = "0")$table
