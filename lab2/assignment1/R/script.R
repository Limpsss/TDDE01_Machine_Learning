# ---- setup-functions ----
# Calculates the mean squared error
mse <- function(y, y_hat) {
  return( mean((y - y_hat)^2) )
}

# ------------ 1 ------------
# ---- read-and-partition-data ----

# Read data
data <- read.csv("./tecator.csv")

# Create training/testing data with ratio 50%/50% using
# partitioning principle from slides
n=dim(data)[1]
set.seed(12345)
id=sample(1:n, floor(n*0.5))
train=data[id,]
test=data[-id,]

# ---- extract-data ----
# Extract the relevant data (Fat, Samples 1..100)
relevant_train <- data.frame(Fat = train$Fat, train[, 2:101])
relevant_test <- data.frame(Fat = test$Fat, test[, 2:101])

# ---- fit-lm ----
# Fit a linear regression model using training data
model_lm <- lm(Fat ~ ., data = relevant_train)

# ---- calculate-mse ----
# Make predictions for the training and testing data.
# Uses the predicted values to calculate MSE
pred_train <- predict(model_lm, newdata = relevant_train)
mse_train <- mse(relevant_train$Fat, pred_train)

pred_test <- predict(model_lm, newdata = relevant_test)
mse_test <- mse(relevant_test$Fat, pred_test)

# ---- print-mse ----
# Prints MSEs for visual clarity
print(mse_train)
print(mse_test)

# ------------ 2 ------------
# ---- training-function ----

library(glmnet)

# Function that takes a data frame with Fat followed by Channel 1..100.
# Returns a glmnet model
# If alpha = 0, returns Ridge
# If alpha = 1, returns LASSO
model_trainer <- function(data, alpha) {
  X <- as.matrix(data[, -1])
  y <- data$Fat

  # alpha = 0 signifies Ridge, alpha = 1 signifies LASSO
  return(glmnet(X, y, alpha = alpha))
}

# ---- training-lasso ----
model_lasso <- model_trainer(relevant_train, 1)

# ------------ 3 ------------

# ---- plot-lasso ----
plot(model_lasso, xvar = "lambda", sign.lambda = 1, label = TRUE)

# ------------ 4 ------------
# ---- training-ridge ----

model_ridge <- model_trainer(relevant_train, 0)

# ---- plot-ridge ----
plot(model_ridge, xvar = "lambda", sign.lambda = 1, label = TRUE)

# ------------ 5 ------------

# ---- cv-function ----
# Trains a model with cross-validation
# If alpha = 0, returns Ridge
# If alpha = 1, returns LASSO
cv_calculator <- function(data, alpha) {
  X <- as.matrix(data[, -1])
  y <- data$Fat

  # alpha = 0 signifies Ridge, alpha = 1 signifies LASSO
  return(cv.glmnet(X, y, alpha = alpha))
}

# ---- training-cv ----
lasso_cv <- cv_calculator(relevant_train, 1)

# ---- plot-cv ----
plot(lasso_cv, sign.lambda = 1)

# ---- variables-chosen-cv ----
# Prints the number of non-zero features for the optimal lambda
# (subtracts one because the intercept is a non-zero)
print(sum(coef(lasso_cv, s = "lambda.min") != 0) - 1)

# ---- scatter ----
# Extract the X matrix from the test data
cv_test_X <- as.matrix(relevant_test[, -1])

# Predict using the cross-validated model using the optimal lambda
predict_cv <- predict(lasso_cv, newx = cv_test_X, s = "lambda.min")

row_range <- 1:nrow(relevant_test)

plot(row_range,
     relevant_test$Fat,
     col = adjustcolor("blue", 0.7),
     xlab="Sample Number",
     ylab="Value",
     lty = 1,
     pch = 16)

points(row_range,
       as.vector(predict_cv),
       col = adjustcolor("red", 0.7),
       lty = 1,
       pch = 16)

legend("bottomright",
       legend = c("Actual Value", "Predicted Value"),
       col = c("blue", "red"),
       pch = 19, cex = 0.8)
