# Load necessary libraries
library(tidyr)
library(dplyr)
library(readr)
library(C50)
library(randomForest)


# Read the dataset
data <- read_csv("absenteeism_1.csv")

# Count number of rows with missing values
missing_rows <- sum(is.na(data))

# Print the result
cat("Number of rows with missing values: ", missing_rows, "\n")

# Delete rows with missing values
clean_data <- na.omit(data)

# Convert Abs_cat to numerical values (0, 1, 2)
lookup_table <- c("Abs_low" = 0, "Abs_Med" = 1, "Abs_High" = 2)
clean_data$Abs_cat <- as.integer(lookup_table[clean_data$Abs_cat])

# Convert Age_cat to factor
clean_data$Age_cat <- as.factor(clean_data$Age_cat)

# Perform one hot encoding on Trans_expense_cat
encoded_trans_expense_cat <- model.matrix(~ Trans_expense_cat - 1, data = clean_data)

# Perform one hot encoding on Dist_to_work
encoded_dist_to_work <- model.matrix(~ Dist_to_work - 1, data = clean_data)

# Perform one hot encoding on Age_cat
encoded_age_cat <- model.matrix(~ Age_cat - 1, data = clean_data)

# Combine encoded columns with the original dataframe
clean_data <- cbind(clean_data[, -which(names(clean_data) %in% c("Trans_expense_cat", "Dist_to_work", "Age_cat"))],
                    encoded_trans_expense_cat,
                    encoded_dist_to_work,
                    encoded_age_cat)

# Convert all columns to factors
clean_data <- clean_data %>% 
  mutate(across(everything(), as.factor))

# Print the cleaned data
print(clean_data)

# Split the dataset into training (70%) and test (30%) sets
set.seed(123)
split_indices <- sample(1:nrow(clean_data), 0.7 * nrow(clean_data))
train_data <- clean_data[split_indices, ]
test_data <- clean_data[-split_indices, ]

# Define the random forest model
model <- randomForest(Abs_cat ~ ., data = train_data,
                      mtry = 3,
                      ntree = 500)


# Make predictions using the test dataset
predictions <- predict(model, test_data)

# Calculate accuracy
accuracy <- sum(predictions == test_data$Abs_cat) / nrow(test_data)

# Calculate precision for Abs_cat=Abs_High
tp <- sum(predictions == "2" & test_data$Abs_cat == "2")
fp <- sum(predictions == "2" & test_data$Abs_cat != "2")
precision <- tp / (tp + fp)

# Print the results
cat("Accuracy of the classification: ", accuracy, "\n")
cat("Precision of the classification for Abs_cat=Abs_High: ", precision, "\n")