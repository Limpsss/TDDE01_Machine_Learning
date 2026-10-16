# ---- Functions ----
library(tree)
missclass=function(X,X1){ # From the lecture slides 1.c
  n=length(X)
  return(1-sum(diag(table(X,X1)))/n)
}

f1 <- function(conf_matrix){
  # In order to calculate F1-score we must first calculate precision and recall.
  # precision = true_positives / (true_positives + false_positives)
  precision <- conf_matrix[2,2] / (conf_matrix[2,2] + conf_matrix[1,2])
  
  # recall = true_positives / (true_positives + false_negatives)
  recall <- conf_matrix[2,2] / (conf_matrix[2,2] + conf_matrix[2,1])
  
  # F1-score is calculated as follows: 
  return(2*(precision*recall)/(precision+recall))
}

# ---- part1 ----
data <- read.csv2("./bank-full.csv")

# Exclude "duration" input variable
data <- data[ , -which(names(data) == "duration")] 

# Convert all character columns to factors to avoid NAs during 
#construction of decision trees
data[] <- lapply(data, function(x) if (is.character(x)) factor(x) else x)

# Code taken from slide 20 of lecture 2a
n=dim(data)[1]
set.seed(12345) 
id=sample(1:n, floor(n*0.4)) 
train=data[id,] 
id1=setdiff(1:n, id)
set.seed(12345) 
id2=sample(id1, floor(n*0.3)) 
valid=data[id2,]
id3=setdiff(id1,id2)
test=data[id3,] 


# ---- fit1 ----
fit1 <- tree(y~., data=train)
plot(fit1)
text(fit1, pretty = 0, cex=0.7)

# ---- fit1-miss ----
#Misclassification for default settings tree
Yfit1_train <- predict(fit1, newdata=train, type="class")
fit1_training_miss <- missclass(train$y, Yfit1_train)
Yfit1_valid <- predict(fit1, newdata=valid, type="class")
fit1_validation_miss <- missclass(valid$y, Yfit1_valid)
print(paste("fit1 training misclassification: ", fit1_training_miss))
print(paste("fit1 validation misclassification: ", fit1_validation_miss))

# ---- fit2 ----
ctrl <- tree.control(nobs <- nrow(train), minsize=7000)
fit2 <- tree(y~., data=train, control=ctrl)
plot(fit2)
text(fit2, pretty = 0, cex=0.7)

# ---- fit2-miss ----
#Misclassification for tree with smallest node size equal to 7000
Yfit2_train <- predict(fit2, newdata=train, type="class")
fit2_training_miss <- missclass(train$y, Yfit2_train)
Yfit2_valid <- predict(fit2, newdata=valid, type="class")
fit2_validation_miss <- missclass(valid$y, Yfit2_valid)
print(paste("Fit2 training misclassification: ", fit2_training_miss))
print(paste("Fit2 validation misclassification: ", fit2_validation_miss))

# ---- fit3 ----
ctrl <- tree.control(nobs <- nrow(train), mindev=0.0005)
fit3 <- tree(y~., data=train, control=ctrl)
plot(fit3, type="u")
text(fit3, pretty = 0, cex=0.5)

# ---- fit3-miss ----
#Misclassification for tree with minimum deviance at 0.0005
Yfit3_train <- predict(fit3, newdata=train, type="class")
fit3_training_miss <- missclass(train$y, Yfit3_train)
Yfit3_valid <- predict(fit3, newdata=valid, type="class")
fit3_validation_miss <- missclass(valid$y, Yfit3_valid)
print(paste("Fit3 training misclassification: ", fit3_training_miss))
print(paste("Fit3 validation misclassification: ", fit3_validation_miss))


# ---- part3 ----
#Create two vectors of length 50 filled with zeros
train_deviances <- rep(0,50)
test_deviances <- rep(0,50)

#Examine trees up to 50 leaves to find optimal tree depth
for (i in 2:50){
  pruned_tree <- prune.tree(fit3, best=i)
  prediction <- predict(pruned_tree, newdata=valid, type="tree")
  train_deviances[i] <- deviance(pruned_tree)
  test_deviances[i] <- deviance(prediction)
}

#scale graph depending on min and max of test and train scores
#first values are skipped as they are not updated in the loop
ylim <- c(min(test_deviances[-1]), max(train_deviances[-1])) 

plot(
  2:50, 
  train_deviances[2:50], 
  type="b", 
  col="red", 
  ylim=ylim, 
  xlab="Number of Leaves", 
  ylab="Deviance")

points(
  2:50, 
  test_deviances[2:50], 
  type="b", 
  col="blue")

legend(
  "topright", 
  legend = c("Training data", "Test data"), 
  col = c("red", "blue"), 
  pch = 19, 
  cex = 0.8)

# ---- opt-leaves ----
optimal_leaves <- which.min(test_deviances[2:50]) + 1
print(paste("The optimal number of leaves is:", optimal_leaves))

# ---- opt-tree ----
optimal_tree <- prune.tree(fit3, best=optimal_leaves)
optimal_tree_predictions <- predict(optimal_tree, newdata=valid, type="class")
plot(optimal_tree, type="u")
text(optimal_tree, pretty = 0, cex = 0.7)


# ---- part4-conf ----
#Get predictions using test data
optimal_tree_predictions <- predict(optimal_tree, newdata=test, type="class")

#Print confusion matrix
conf_matrix <- table(Actual = test$y, Predicted = optimal_tree_predictions)
conf_matrix

# From the confusion matrix we can see that the majority of all calls that
# are made are unsuccessful. Only about 291/11812 or about 2.5% are
# successful. 

# ---- part4-acc ----
# This gives us a total misclassification of only about 10%
miss <- missclass(test$y, optimal_tree_predictions)
accuracy <- 1-miss
print(paste(("Accuracy for optimal tree: "), accuracy))

# ---- part4-f1 ----
# In our scenario the majority of calls are unsuccessful which means it might
# be more appropriate to use F1-score as a measure for predictive power.
f1_score <- f1(conf_matrix)
print(paste(("F1 Score for optimal tree: "), f1_score))

# The F1-score is very low which indicates that the decision tree is not very
# good at predicting cases of "yes".
# For our tree the F1-score should be the preferred as the dataset is very
# unbalanced (there is a lot more "no" than "yes")


# ---- part5 ----
# There is no way in the "tree" library to apply a loss matrix directly so we
# have to apply the loss after prediction.
optimal_tree_predictions <- predict(optimal_tree, newdata=test)

# Calculate predicted loss
loss <- matrix(c(0, 5, 1, 0), nrow = 2)
predicted_loss <- optimal_tree_predictions %*% loss #matrix multiplication

# Update predictions according to minimum predicted loss
optimal_tree_predictions <- levels(test$y)[apply(predicted_loss, 1, which.min)]

#Print confusion matrix
conf_matrix <- table(Actual = test$y, Predicted = optimal_tree_predictions)
conf_matrix

# ---- part5-acc
miss <- missclass(test$y, optimal_tree_predictions)
accuracy <- 1-miss
print(paste(("Accuracy for tree with loss matrix: "), accuracy))

# ---- part5-f1
f1_score <- f1(conf_matrix)
print(paste(("F1 Score for tree with loss matrix: "), f1_score))

# In the classification with the loss matrix the accuracy is slightly lower
# (87.3% compared to 89.2%) but the F1-score is substantially better
# (48.6% compared to 28.5%).

# With the loss matrix the decision tree will try harder to predict "yes"
# correctly at the cost of slightly lower accuracy. Our dataset is very
# unbalanced which makes the F1-score a more important statistic to
# optimize for.


# ---- part6 ----
TPR <- function(conf_matrix){
  conf_matrix
  true_positives <- conf_matrix[4]
  false_negatives <- conf_matrix[2]
  return (true_positives / (true_positives+false_negatives))
}

FPR <- function(conf_matrix){
  false_positives <- conf_matrix[3]
  true_negatives <- conf_matrix[1]
  return (false_positives / (false_positives+true_negatives))
}

# Fit logistic regression
glm_fit <- glm(y~., data = train, family = "binomial")

# Initialize 
tree_tpr <- numeric()
tree_fpr <- numeric()
glm_tpr <- numeric()
glm_fpr <- numeric()

# Calculate TPR and FPR for optimal tree and glm fit for different values of "pi"
for (pi in seq(from=0.05, to=0.95, by=0.05)){
  tree_probabilities <- predict(optimal_tree, newdata=test)
  tree_predictions <- ifelse(tree_probabilities[, "yes"] > pi, "yes", "no")
  tree_conf_matrix <- table(Actual = test$y, Predicted = tree_predictions)
  # Calculate and add to result vectors
  tree_tpr <- c(tree_tpr, TPR(tree_conf_matrix))
  tree_fpr <- c(tree_fpr, FPR(tree_conf_matrix))
  
  glm_probabilities <- predict(glm_fit, newdata=test, type="response")
  glm_predictions <- ifelse(glm_probabilities > pi, "yes", "no")
  glm_conf_matrix <- table(Actual = test$y, Predicted = glm_predictions)
  # Calculate and add to result vectors
  glm_tpr <- c(glm_tpr, TPR(glm_conf_matrix))
  glm_fpr <- c(glm_fpr, FPR(glm_conf_matrix))
}

plot(
  tree_fpr, 
  tree_tpr, 
  main="ROC-curves",
  xlab="False Positive Rate", 
  ylab="True Positive Rate", 
  col="red", 
  type="b", 
  xlim=c(0,0.7), 
  ylim=c(0,0.9))

points(glm_fpr, glm_tpr, type="b") # type="b" draws a line between each point

legend(
  "topleft", 
  legend = c("Decision tree", "Logistic regression"), 
  col = c("red", "black"), 
  pch = 19, 
  cex = 0.8)



