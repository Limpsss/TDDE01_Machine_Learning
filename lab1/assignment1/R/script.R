# ---- setup-functions ----
# Function to calculate missclassification.
# From the lecture slides 1.c
missclass=function(X,X1){
  n=length(X)
  return(1-sum(diag(table(X,X1)))/n)
}

# ------------ 1 ------------
# ---- read-and-partition-data ----
library(kknn)

# Read data
data <- read.csv("./optdigits.csv", header = FALSE)

# Create training/validation/testing data with ratio 50%/25%/25% using
# partitioning principle from slides
n=dim(data)[1]
set.seed(12345)
id=sample(1:n, floor(n*0.5))
train=data[id,]

id1=setdiff(1:n, id)
set.seed(12345)
id2=sample(id1, floor(n*0.25))
valid=data[id2,]

id3=setdiff(id1,id2)
test=data[id3,]

# Rename the column name of the true value in the data to avoid confusion
names(train)[names(train) == "V65"] <- "true"
names(valid)[names(valid) == "V65"] <- "true"
names(test)[names(test) == "V65"] <- "true"

# Convert y-values in training data to factors.
# This makes sure that the model predicts labels and not floats
train$true <- as.factor(train$true)

# ---- train-models ----

# Train the kknn model and evaluate performance on training data
model_train <- kknn(train$true ~ ., train, train, kernel = "rectangular", k = 30)

# Train the kknn model and evaluate performance on testing data
model_test <- kknn(train$true ~ ., train, test, kernel = "rectangular", k = 30)

# ---- get-predictions ----

# Extract the predicted values for training data
prediction_training <- fitted(model_train)

# Extract the predicted values for testing data
prediction_testing <- fitted(model_test)

# ---- show-confusion-matrix-1 ----
# Create confusion matrices for evaluation on training data and on testing data
table(Actual = train$true, Predicted = prediction_training)

# ---- show-confusion-matrix-2 ----
table(Actual = test$true, Predicted = prediction_testing)


# ---- show-misclassifications ----
# Calculate missclassification
misclass_training <- missclass(train$true, prediction_training)
print(misclass_training)

misclass_testing <- missclass(test$true, prediction_testing)
print(misclass_testing)

# ------------ 3 ------------
# ---- print-heatmaps-1 ----
# Specify how confident each prediction is
confidence <- apply(model_train$prob, 1, max)

# Create a data frame containing the important values
df_confidence <- data.frame(
  id = as.character(rownames(train)),
  true_label = train$true,
  predicted = prediction_training,
  confidence = confidence
)

# Exclude all rows where the actual and predicted value are not 8
df_confidence <- df_confidence[df_confidence$true_label == "8" & df_confidence$predicted == "8", ]

# Sort descending by level of confidence
df_confidence <- df_confidence[order(-df_confidence[, "confidence"]), ]

# Pick the two rows where the model is the most confident in the prediction
easy_classify <- head(df_confidence, 2)

# Pick the three rows where the model is the least confident in the prediction
hard_classify <- tail(df_confidence, 3)

# Print heatmap for the easy to classify rows
for (i in 1:nrow(easy_classify)) {
  v <- as.numeric(train[easy_classify$id[i], 1:64])
  m <- matrix(v, 8, 8, TRUE)
  h <- heatmap(m, NA, NA)
  h
}

# ---- print-heatmaps-2 ----

# Print heatmap for the hard to classify rows
for (i in 1:nrow(hard_classify)) {
  v <- as.numeric(train[hard_classify$id[i], 1:64])
  m <- matrix(v, 8, 8, TRUE)
  heatmap(m, NA, NA)
}

# ------------ 4 ------------
# ---- print-plot-1 ----

# Sets up data frame to hold data from evaluating models with k-values 1..30
k_values <- 1:30
classifier_values <- data.frame(
  k = integer(length(k_values)),
  train_error = numeric(length(k_values)),
  valid_error = numeric(length(k_values))
)

# Evaluate each model. Save missclassification error for training data
# and validation data in the data frame classifier_values
for (i in k_values) {
  k_model_train <- kknn(train$true ~ ., train, train, kernel = "rectangular", k = i)
  k_model_valid <- kknn(train$true ~ ., train, valid, kernel = "rectangular", k = i)
  k_prediction_train <- fitted(k_model_train)
  k_prediction_valid <- fitted(k_model_valid)
  k_misclass_train <- missclass(train$true, k_prediction_train)
  k_misclass_valid <- missclass(valid$true, k_prediction_valid)
  classifier_values$k[i] <- i
  classifier_values$train_error[i] <- k_misclass_train
  classifier_values$valid_error[i] <- k_misclass_valid
}

# Range needed to fit all values in the plot
plot_y_range <- range(c(classifier_values$train_error, classifier_values$valid_error))

# Plot the missclassification errors
matplot(classifier_values$k, classifier_values[, c("train_error", "valid_error")],
        type = "o", lty = c(1, 1), pch = c(16, 16), col = c("blue", "orange"),
        xlab = "K-values", ylab = "Misclassification Error",
        main = "Training and Validation Misclassification Errors")

# Add a legend for visual clarity
legend("bottomright",
       legend = c("Training Error", "Validation Error"),
       col = c("blue", "orange"),
       lty = c(1, 1),
       pch = c(16, 16))

# => gives optimal k of k = 4

# Trains a model with k = 4 and evalutes on test data
# and calculates the misclassification error
model_k_4_test <- kknn(train$true ~ ., train, test, kernel = "rectangular", k = 4)
k_4_prediction_test = fitted(model_k_4_test)
k_4_misclass_test <- missclass(test$true, k_4_prediction_test)

# ---- optimal-k-printouts ----
# For comparision purposes, creates a column with all missclassification
# errors for k = 4 in the order - training, validation, testing
k_4_misclasses <- c(classifier_values$train_error[4],
                     classifier_values$valid_error[4],
                     k_4_misclass_test)

print(k_4_misclasses)

# ------------ 5 ------------
# ---- print-plot-2 ----

cross_entropies <- c()

# Evaluate each model with cross-entropy as error
for (i in k_values) {
  k_model_valid <- kknn(train$true ~ ., train, valid, kernel = "rectangular", k = i)

  # Gets the index for the column with the probability for the true value from
  # the probability table inside the model
  true_value_prob_indices <- match(valid$true, colnames(k_model_valid$prob))
  no_of_rows <- 1:nrow(k_model_valid$prob)

  # Extracts all the true value probabilities from the probability table
  true_value_probs <- k_model_valid$prob[cbind(no_of_rows, true_value_prob_indices)]

  # Logs each value according to the formula for cross-entropy
  # Adds a tiny constant to each value to avoid issues with log(0),
  # which is undefined
  logs <- log(true_value_probs + 1e-15)

  # Calculates the total error by taking the negative average of the summation
  # of all logs, according to the formula for cross-entropy
  total_error <- -(sum(logs) / length(logs))
  cross_entropies[i] = total_error
}

# Plot the cross-entropies
plot(k_values, cross_entropies,
     type = "o", lty = 1, pch = 16, col = "blue",
     xlab = "K-values", ylab = "Error as Cross-Entropy",
     main = "Validation Error as Cross-Entropy")
