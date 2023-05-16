# Load necessary libraries
library(cluster)
library(ggplot2)
library(dplyr)
library(readr)
library(factoextra)

# Read the dataset
data <- read_csv("absenteeism_0.csv")

# Delete rows with missing values
data_clean <- na.omit(data)

# Save abs_cat for later comparison
abs_cat <- data_clean$Abs_cat

# Remove abs_cat from clustering data
data_clean$Abs_cat <- NULL

# Normalize the data
data_scaled <- scale(data_clean)

# Perform hierarchical clustering
hclust_obj <- hclust(dist(data_scaled), method = "ward.D2")
hclust_groups <- cutree(hclust_obj, k = 3)

# Perform K-means clustering
set.seed(123)
kmeans_obj <- kmeans(data_scaled, centers = 3)
kmeans_groups <- kmeans_obj$cluster

# Compare the clusters with Abs_cat
data_clean$Abs_cat <- abs_cat
data_clean$cluster_hierarchical <- as.factor(hclust_groups)
data_clean$cluster_kmeans <- as.factor(kmeans_groups)

# Plot the hierarchical clustering dendrogram
fviz_dend(hclust_obj, k = 3, cex = 0.5)

# Plot the K-means clustering results
fviz_cluster(kmeans_obj, data = data_scaled)

# Print the centroid of each K-means cluster
cat("Centroids of K-means clusters:\n")
print(kmeans_obj$centers)

# Compare clusters with Abs_cat
table(data_clean$Abs_cat, data_clean$cluster_hierarchical)
table(data_clean$Abs_cat, data_clean$cluster_kmeans)
