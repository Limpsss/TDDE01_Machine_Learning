# -------- PART 1 --------

library(ggplot2)
library(caret)

data <- read.csv("communities.csv")
target <- data[ , "ViolentCrimesPerPop"]
features <- data[ , -which(names(data) == "ViolentCrimesPerPop")]

scaler <- preProcess(features)
features_s <- predict(scaler, features)

S <- cov(features_s) # Sample covariance matrix
eigen <- eigen(S)
eigen_values <- eigen$values

PC_variance <- eigen_values / sum(eigen_values) * 100 # not needed?
sprintf("%2.3f", PC_variance)

# Number of components to obtain 95% variance
n <- which(cumsum(PC_variance) >= 95)[1]
n

sprintf("PC1: %.2f%%", PC_variance[1])
sprintf("PC2: %.2f%%", PC_variance[2])

# -------- PART 2 --------

pca <- princomp(features_s)
pc1_loadings <- pca$loadings[, 1]

# ---- traceplot ----
plot(
  pc1_loadings,
  main="Traceplot PC1",
  ylab = "Feature loading Value",
  xlab = " Feature index",
  col = "blue",
  pch = 19
)
# ---- ! ----

# A vector of the index of highest loading in decreasing order
top_indices <- order(abs(pc1_loadings), decreasing = TRUE)

# Top 5 features that contribute the most
top_features <- data.frame(
  Feature = names(pc1_loadings)[top_indices[1:5]],
  Loading = abs(pc1_loadings[top_indices[1:5]])
)
rownames(top_features) <- NULL
top_features

pc1_scores <- pca$scores[, 1]
pc2_scores <- pca$scores[, 2]

# PC scores (PC1, PC2) and target
df <- data.frame(
  PC1 = pc1_scores,
  PC2 = pc2_scores,
  ViolentCrimesPerPop = target
)

# ---- PC-scores-plot ----
# ggplot of PC scores (PC1, PC2)
ggplot(df, aes(x = PC1, y = PC2, color = ViolentCrimesPerPop)) +
  geom_point() +
  scale_color_gradient(low = "blue", high = "red") +
  labs(title = "PC scores", x = "PC1", y = "PC2") +
  theme_minimal()
# ---- ! ----

# -------- PART 3 --------

# Partitioning principle from slides
n=dim(data)[1]
set.seed(12345)
id=sample(1:n, floor(n*0.5))
train=data[id,]
test=data[-id,]

# Scaling data
scaler <- preProcess(train)
train_s <- predict(scaler, train)
test_s <- predict(scaler, test)

mse <- function(y, y_hat) {
  return( mean((y - y_hat)^2) )
}

# Linear regression model on predictor ViolentCrimesPerPop
model <- lm(train_s$ViolentCrimesPerPop ~ . -1, data=train_s)

# Predictions for train and test data
train_preds <- predict(model, newdata = train_s)
test_preds <- predict(model, newdata = test_s)

# MSE scores for train and test data
mse_train <- mse(train_s$ViolentCrimesPerPop, train_preds)
mse_test <- mse(test_s$ViolentCrimesPerPop, test_preds)

# -------- PART 4 --------

cost_function <- function(theta, X, y) {
  n <- length(y)
  y_hat <- X %*% theta
  return (mse(y, y_hat))
}

optimize_theta <- function() {
  # Starting point
  theta_start <- rep(0, ncol(X_train))

  # Each iteration
  obj_function <- function(theta_hat) {

    # Calculating MSE for train and test on each iteration
    train_error <- mse(y_train, X_train %*% theta_hat)
    test_error <- mse(y_test, X_test %*% theta_hat)

    # Add current iteration errors to global error vectors
    train_errors <<- c(train_errors, train_error)
    test_errors <<- c(test_errors, test_error)

    return (cost_function(theta_hat, X_train, y_train))
  }

  optim_result <- optim(
    par = theta_start,
    fn = obj_function,
    method = "BFGS",
  )

  return (optim_result = optim_result)
}

X_train <- as.matrix(train_s[, -which(names(train_s) == "ViolentCrimesPerPop")])
y_train <- train_s$ViolentCrimesPerPop
X_test <- as.matrix(test_s[, -which(names(train_s) == "ViolentCrimesPerPop")])
y_test <- test_s$ViolentCrimesPerPop

train_errors <- c()
test_errors <- c()

res <- optimize_theta()
min_test_error <- min(test_errors)
min_test_index <- which.min(test_errors)

# ---- learning-curve-plot ----
# Plot training errors
plot(
  train_errors,
  type = "l",
  col = "blue",
  ylab = "MSE",
  xlab = "Iteration",
  main = "Learning curve (Training vs Testing)",
  ylim = c(0.2, 0.7),
  xlim = c(500, 10000),
  lwd = 2
)

# Test errors line
lines(
  test_errors,
  type = "l",
  col = "red",
  lwd = 2
)

# Optimal test error point
points(
  x = min_test_index,
  y = min_test_error,
  col = "green",
  pch = 19,
  cex = 1.2
)

legend(
  "topright",
  legend = c("Training Error", "Test Error", "Optimal Test Error"),
  lty = c(1, 1, 0),
  col = c("blue", "red", "green"),
  pch = c(NA, NA, 19),
)

optimal_model_test_mse <- min_test_error
optimal_model_train_mse <- train_errors[min_test_index]

# print("Training and test error for optimal model:")
# sprintf("Test error: %f", min_test_error)
# sprintf("Train error: %f", train_errors[min_test_index])


