# Load necessary libraries
library(data.table)
library(dplyr)
library(caret)
library(ROCR)
library(foreign)

# Start measuring runtime
start_time <- Sys.time()

# Load the training and testing data from the ARFF files
training_data <- read.arff("/Users/carlostorres/Documents/Python Project/Data Sets/Annthyroid/Annthyroid_02_v10.arff")
testing_data <- read.arff("/Users/carlostorres/Documents/Python Project/Data Sets/Annthyroid/Annthyroid_withoutdupl_norm_07.arff")

# Check the unique values in the target column 'outlier' in the training data
print("Unique values in training data:")
print(table(training_data$outlier))

# Check the unique values in the target column 'outlier' in the testing data
print("Unique values in testing data:")
print(table(testing_data$outlier))

# Convert the target variable to a factor if it is not already
training_data$outlier <- as.factor(training_data$outlier)
testing_data$outlier <- as.factor(testing_data$outlier)

# Ensure both levels of the target variable are present in the training data
if(length(unique(training_data$outlier)) < 2) {
  stop("The training data must have at least two unique values for the target variable")
}

# Building a logistic regression model
model <- train(outlier ~ ., data = training_data, method = "glm", family = "binomial")

# Making predictions on the test set
predictions <- predict(model, testing_data, type = "raw")

# Calculating the accuracy
conf_matrix <- confusionMatrix(predictions, testing_data$outlier)
accuracy <- conf_matrix$overall['Accuracy']
print(paste("Accuracy: ", accuracy))

# Making probability predictions on the test set for AUC calculation
prob_predictions <- predict(model, testing_data, type = "prob")[,2]

# Calculating the AUC
pred <- prediction(prob_predictions, testing_data$outlier)
perf <- performance(pred, measure = "auc")
auc <- perf@y.values[[1]]
print(paste("AUC: ", auc))

# End measuring runtime
end_time <- Sys.time()
runtime <- end_time - start_time
print(paste("Runtime: ", as.numeric(runtime), "seconds"))