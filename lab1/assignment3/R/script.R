
# ---- Functions ----
missclass=function(X,X1){ # From the lecture slides 1.c
  n=length(X)
  return(1-sum(diag(table(X,X1)))/n)
}

get_predictions <- function(model, threshold) {
  probabilities <- predict(model, type = "response") # Type "response" returns predicted probabilities
  return(ifelse(probabilities >= threshold, 1, 0)) 
}

# The equation for the decision boundary can be derived by solving the 
# probabilistic equation of the estimated model in regards to x2:
# x2 = (log(-(r/(r-1))) - β2*x1 + β0)/ β1
calculate_decision_boundary <- function(model, x1, r) {
  coefficients <- coef(model)
  b  <- coefficients[1]
  w1 <- coefficients[2]
  w2 <- coefficients[3]
  x2 <- (log(r / (1 - r)) - b - w1 * x1) / w2
  return(x2)
}

get_colors <- function(x) {
  return(ifelse(x == 1, "red", "black"))
}

# ---- Read-data ----
data <- read.csv("./pima-indians-diabetes.csv", header=F)
colnames(data) <- c(
  "times_pregnant", 
  "plasma_glucose_concentration", 
  "diastolic_blood_pressure", 
  "triceps_skinfold_thickness", 
  "2-hour_serum_insulin", 
  "body_mass_index", 
  "diabetes_pedigree_function", 
  "age", 
  "diabetes")


# ---- Part1 ----
#Plot of Plasma glucose concentration on Age, colored by Diabetes levels.
plot(data$plasma_glucose_concentration, data$age, col=get_colors(data$diabetes), main="Real data", xlab="Plasma Glucose Concentration", ylab="Age")
legend("topright", legend = c("Diabetes", "Not diabetes"), col = c("red", "black"), pch = 19, cex = 0.8)


# ---- Part2 ----
# Fit logistic regression
model1 <- glm(data$diabetes ~ data$plasma_glucose_concentration + data$age, data = data, family = "binomial")

# Get prediction
predicted_diabetes <- get_predictions(model1, 0.5)

# Plot predictions
plot(data$plasma_glucose_concentration, data$age, col=get_colors(predicted_diabetes), main="Predicted data", xlab="Plasma Glucose Concentration", ylab="Age")
legend("topright", legend = c("Diabetes", "Not diabetes"), col = c("red", "black"), pch = 19, cex = 0.8)

# ---- misclassification ----
misclassification_model1 <- missclass(data$diabetes, predicted_diabetes) # Calculate the misclassification of the predictions
misclassification_model1


# ---- Part3 ----
plot(data$plasma_glucose_concentration, data$age, col=get_colors(predicted_diabetes), main="Predicted data with decision boundary", xlab="Plasma Glucose Concentration", ylab="Age")
legend("topright", legend = c("Diabetes", "Not diabetes"), col = c("red", "black"), pch = 19, cex = 0.8)

# Generate 100 equally spaced points covering the data range
x_points <- seq(min(data$plasma_glucose_concentration), max(data$plasma_glucose_concentration), length.out = 100)
y_points <- calculate_decision_boundary(model1, x_points, 0.5)
lines(x_points, y_points, lwd = 1) # Draw line covering generated points


# ---- Part4 ----
predicted_probabilities <- predict(model1, type = "response") # Type "response" returns predicted probabilities
par(mfrow = c(1, 2))  

# r=0.2
predicted_diabetes <- ifelse(predicted_probabilities > 0.2, 1, 0)
plot(data$plasma_glucose_concentration, data$age, col=get_colors(predicted_diabetes), main="r=0.2", xlab="Plasma Glucose Concentration", ylab="Age")

# r=0.8
predicted_diabetes <- ifelse(predicted_probabilities > 0.8, 1, 0)
plot(data$plasma_glucose_concentration, data$age, col=get_colors(predicted_diabetes), main="r=0.8", xlab="Plasma Glucose Concentration", ylab="Age")

par(mfrow = c(1, 1)) 

# ---- Part5 ----
# Feature expansion
x1 <- data$plasma_glucose_concentration
x2 <- data$age
data$z1 <- x1^4
data$z2 <- (x1^3)*x2
data$z3 <- (x1^2)*(x2^2)
data$z4 <- x1*(x2^3)
data$z5 <- x2^4

# Train a new model on new features
model2 <- glm(diabetes ~ plasma_glucose_concentration + age + z1 + z2 + z3 + z4 + z5, data = data, family = "binomial")
predicted_diabetes <- get_predictions(model2, 0.5)

plot(data$plasma_glucose_concentration, data$age, col=get_colors(predicted_diabetes), main="Predicted data with expanded features", xlab="Plasma Glucose Concentration", ylab="Age")
legend("topright", legend = c("Diabetes", "Not diabetes"), col = c("red", "black"), pch = 19, cex = 0.8)

misclassification_model2 <- missclass(data$diabetes, predicted_diabetes)

# ---- misclassification2 ----
print(paste("The original model has a misclassification of ", misclassification_model1, " or about ", round(misclassification_model1 * 100, 2), "%"))
print(paste("The model with expanded features has a misclassification of ", misclassification_model2, " or about ", round(misclassification_model2 * 100, 2), "%"))
diff <- (misclassification_model1)-(misclassification_model2)
print(paste("The model with expanded features is around", round(diff * 100, 2), "% better than the original"))

