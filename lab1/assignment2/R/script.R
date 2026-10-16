## -------- PART 1 --------

library(caret)

data <- read.csv("./parkinsons.csv")
# Create training/testing data with ratio 60%/40% using
# partitioning principle from slides
n=dim(data)[1]
set.seed(12345)
id=sample(1:n, floor(n*0.6))
train=data[id,]
test=data[-id,]
relevant_train_data <- data.frame(motor_UPDRS=train$motor_UPDRS, train[, 7:22])
relevant_test_data <- data.frame(motor_UPDRS=test$motor_UPDRS, test[, 7:22])

# Scale data from slides
scaler=preProcess(relevant_train_data)
train_s=predict(scaler, relevant_train_data)
test_s=predict(scaler, relevant_test_data)

## -------- PART 2 --------

# Fitting a linear regression model with 0 intercept
model <- lm(motor_UPDRS ~ . -1, data=train_s)

# Calculates the mean squared error
mse <- function(y, y_hat) {
  return( mean((y - y_hat)^2) )
}

preds_train <- predict(model, newdata = train_s)
mse_train <- mse(train_s$motor_UPDRS, preds_train)

preds_test <- predict(model, newdata = test_s)
mse_test <- mse(test_s$motor_UPDRS, preds_test)

## -------- PART 3 --------

# Log likelihood function:
log_likelihood <- function(theta, sigma, X, y) {
  # Calculate the SSE
  y_hat <- X %*% theta
  SSE <- sum((y - y_hat)^2)

  N <- length(y)
  # log-likelihood
  ll <- (-N/2 * log(2 * pi * sigma^2)) - (SSE / (2 * sigma^2))
  return(ll)
}

# Ridge function:
ridge <- function(theta, sigma, lambda, X, y) {
  ll <- log_likelihood(theta, sigma, X, y)
  penalty <- lambda * sum(theta^2)

  return(-ll + penalty)
}

# RidgeOpt function:
ridgeOpt <- function(lambda, X, y) {
  # Objective function for optim
  obj_function <- function(params) {
    params_len <- length(params)
    theta <- params[1:(params_len - 1)]
    log_sigma <- params[params_len]
    sigma <- exp(log_sigma) # To handle sigma 0 or < 0
    return(ridge(theta, sigma, lambda, X, y))
  }
  # Start with theta and sigma as 0
  params_to_optimize <- rep(0, 17)
  opt_result <- optim(
    par = params_to_optimize,
    fn = obj_function,
    method = "BFGS"
  )

  return(opt_result)
}

# Degrees of Freedom function
DF <- function(lambda, X) {
  # Identity matrix (p x p)
  I <- diag(ncol(X))
  # Hat Matrix Ridge = X * (X^T * X + lambda * I)^-1 * X^T
  H_Ridge <- X %*% solve(( t(X) %*% X + lambda * I )) %*% t(X)
  df <- sum(diag(H_Ridge))
  return(df)
}

## -------- PART 4 --------

X_train <- as.matrix(train_s[, -which(names(train_s) == "motor_UPDRS")])
y_train <- train_s$motor_UPDRS
X_test <- as.matrix(test_s[, -which(names(test_s) == "motor_UPDRS")])
y_test <- test_s$motor_UPDRS

for (lambda in c(1, 100, 1000)) {
  ridge_model <- ridgeOpt(lambda, X_train, y_train)
  theta_hat <- ridge_model$par[1:16]

  y_hat_train <- X_train %*% theta_hat
  y_hat_test <- X_test %*% theta_hat

  #print(sprintf("Train data - lambda: %d, mse: %f", lambda, mse(y_train, y_hat_train)))
  #print(sprintf("Test data - lambda: %d, mse: %f", lambda, mse(y_test, y_hat_test)))

  cat("\n")
}

for (lambda in c(1, 100, 1000)) {
  df <- DF(lambda, X_train)
  #print(sprintf("Lambda: %d, df: %f", lambda, df))
}

