# Name : Naveen Mathews Renji | CWID : 20016323 

# Import Libraries
library(caret)
library(randomForest)

# Load dataset from csv file
df <- read.csv('/Users/naveenrenji/Stevens/CS513/hw/hw6_C5andRF/breast-cancer-wisconsin.csv')

# Drop the 'Sample' column
df <- df[,-1]

# Replace missing values with mean and convert 'F6' column to numeric datatype
df$F6 <- as.numeric(replace(df$F6, df$F6 == '?', NA))
df$F6[is.na(df$F6)] <- mean(df$F6, na.rm = TRUE)

# Convert 'Class' column to factor datatype
df$Class <- as.factor(df$Class)

# Separate the target variable and feature variables
y <- df$Class
X <- df[,-9]

# Convert the target variable to binary classification problem
y <- ifelse(y == 2, 0, 1)

# Split the data into training and testing sets
train_index <- createDataPartition(y, p = 0.7, list = FALSE)
train <- df[train_index,]
test <- df[-train_index,]

# Train a decision tree classifier on the training set
dTree <- train(Class ~ ., data = train, method = 'C5.0')

# Train a random forest classifier on the training set
rf <- randomForest(Class ~ ., data = train)

# Evaluate the performance of the decision tree model on the testing set
y_pred_dt <- predict(dTree, newdata = test)

# Evaluate the performance of the random forest model on the testing set
y_pred_rf <- predict(rf, newdata = test)

# Results
print("Results for C5 classifier")
confusionMatrix(y_pred_dt, test$Class)

print("Results for Random Forest Methodology")
confusionMatrix(y_pred_rf, test$Class)
