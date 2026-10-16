# The following code is based on the given template from the lab assignment
# ---- init ----
library(neuralnet)
set.seed(1234567890)

plot_nn <- function(train, test, nn)
{
  # Plot of the training data (black), test data (blue), and predictions (red)
  plot(train, cex=2,, xlab="x", ylab="Sin(x)")
  points(test, col = "blue", cex=1)
  points(test[,1], predict(nn,test), col="red", cex=1)
  
  legend(
    "bottomleft", 
    legend = c("Training data", "Test data", "Predictions"), 
    col = c("black", "blue", "red"), 
    pch = 19, 
    cex = 0.8)
}

values <- runif(500, 0, 10)
data <- data.frame(values, Sin=sin(values))
train <- data[1:25,] # Training
test <- data[26:500,] # Test

# Random initialization of the weights in the interval [-1, 1]
# each hidden unit has an input variable and bias and each requires a weight
# -> 20 weights
# output unit has each input unit + bias
# -> 11 weights
# --> 31 weights
winit <- runif(31, -1, 1)

# ---- part1 ----
nn <- neuralnet(
  Sin~., 
  data = train, 
  hidden = 10, 
  startweights = winit)
plot_nn(train, test, nn)


# ---- part2-1 ----
# Linear activation function
nn_linear <- neuralnet(
  Sin~., 
  data = train,
  hidden = 10, 
  act.fct = function(x) { x }, 
  startweights = winit)
plot_nn(train, test, nn_linear)

# ---- part2-2 ----
# ReLU activation function
nn_relu <- neuralnet(
  Sin~., 
  data = train, 
  hidden = 10, 
  act.fct = function(x) { ifelse(x > 0, x, 0) }, 
  startweights = winit)
plot_nn(train, test, nn_relu)

# ---- part2-3 ----
# Softplus activation function
nn_softplus <- neuralnet(
  Sin~., 
  data = train, 
  hidden = 10, 
  act.fct = function(x) { log(1 + exp(x)) },
  startweights = winit)
plot_nn(train, test, nn_softplus)


# ---- part3 ----
values <- runif(500, 0, 50)
data <- data.frame(values, Sin=sin(values))

plot(data, ylim = c(-12, 2), xlab="x", ylab="Sin(x)")
points(data[, 1], predict(nn, data[1]), col = "red")

legend(
  "bottomleft", 
  legend = c("Actual", "Predicted"), 
  col = c("black", "red"), 
  pch = 19, 
  cex = 0.8)


# ---- part4 ----
# The model fits to the sine curve very well for values of x inside the expected
# range of 0-10. As x grows beyond x=10 the predictions deviate towards a
# constant x=~-10.

# This is because as the input grows the activation functions for the hidden
# units starts to saturate. Our NN uses a sigmoid activation function. As the 
# value of X for a sigmoid increases its output starts to approach "1" while
# when it decreases it approaches "0".

nn$weights
# By looking at the learned weights we can see that some of the weights are 
# relatively large which means that the input to the activation functions also 
# will be large. Due to the nature of the sigmoid activation
# function, large inputs will quickly go towards either 0 or 1. When this
# happens the curve flattens.


# ---- part5 ----
values <- runif(500, 0, 10)
data <- data.frame(Sin = sin(values), values)

nn <- neuralnet(
  values~., 
  data = data, 
  hidden = 10, 
  startweights = winit,
  threshold = 0.1)

plot(
  data, 
  cex=2, 
  col="blue",
  xlab="Sin(x)",
  ylab="x")

points(
  data[,1], 
  predict(nn, data), 
  col="red", 
  cex=1)

legend(
  "bottomleft", 
  legend = c("Training data", "Predictions"), 
  col = c("blue", "red"), 
  pch = 19, 
  cex = 0.8)