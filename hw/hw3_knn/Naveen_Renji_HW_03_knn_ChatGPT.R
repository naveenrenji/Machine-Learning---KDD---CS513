# Name : Naveen Mathews Renji | CWID : 20016323

# Import Libraries
library(tidyverse)
library(class)

# Load the dataset
df <- read.csv("breast-cancer-wisconsin.csv")

head(df)

num_rows <- nrow(df)
print(num_rows)

# Taking care of missing values
df[df == '?'] <- NA
df$F6 <- as.numeric(df$F6)

df <- df %>%
  mutate(F6 = ifelse(is.na(F6), mean(df$F6, na.rm = TRUE), F6))

sum(is.na(df))

# Creating KNN Model and classifying the training data
X <- df %>% select(-ID, -Class) %>% as.matrix()
y <- df$Class %>% as.factor()

set.seed(42)
train_indices <- sample(seq_len(nrow(df)), size = 0.7*nrow(df))

X_train <- X[train_indices,]
X_test <- X[-train_indices,]
y_train <- y[train_indices]
y_test <- y[-train_indices]

neighbours_size <- c(3, 5, 10)
for (i in neighbours_size) {
  knn <- knn(train = X_train, test = X_test, cl = y_train, k = i)
  Y_pred <- as.factor(knn)
  print(paste("The Classifiction models for KNN methodology for k Value =", i))
  print(paste("The Score is -", mean(Y_pred == y_test)))
  print("The Confusion Matrix is -")
  print(confusionMatrix(Y_pred, y_test))
  print("The Classification Report Matrix is -")
  print(classSummary(Y_pred, y_test))
}
