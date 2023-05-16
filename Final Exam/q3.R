library(tidyverse)
library(caret)

# Read the CSV dataset
data <- read.csv("absenteeism_1.csv")

# Delete rows with missing values
data_clean <- na.omit(data)

# Convert columns to appropriate data types
data_clean <- data_clean %>%
  mutate_at(vars(Month_of_absence, Day_of_the_week, Social_drinker, Social_smoker, Pet),
            as.numeric) %>%
  mutate_at(vars(Trans_expense_cat, Dist_to_work, Age_cat, Abs_cat),
            as.factor)

# Set seed for reproducibility
set.seed(123)

# Split the dataset into training and testing sets
splitIndex <- createDataPartition(data_clean$Abs_cat, p = 0.7, list = FALSE)
train_data <- data_clean[splitIndex, ]
test_data <- data_clean[-splitIndex, ]

# Train the model
model_cart <- train(Abs_cat ~ ., data = train_data, method = "rpart")

# Make predictions and calculate accuracy
predictions_cart <- predict(model_cart, newdata = test_data)
confusionMatrix_cart <- confusionMatrix(predictions_cart, test_data$Abs_cat)
accuracy_cart <- confusionMatrix_cart$overall["Accuracy"]

cat("CART Accuracy: ", accuracy_cart, "\n")

stats_by_class <- as.data.frame(confusionMatrix_cart$byClass)

precision_abs_high <- stats_by_class["Class: Abs_High", "Pos Pred Value"]
cat("Precision for Abs_cat=Abs_High: ", precision_abs_high, "\n")
