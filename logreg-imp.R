library("dplyr")
library("caTools")
library("ROCR")
library("InformationValue")
my_data <- read.csv("binary.csv")
X <- as.matrix(my_data[, names(my_data) != "admit"])
X <- cbind(rep(1, nrow(X)), X)
y <- as.matrix(my_data$admit)

sigmoid <- function(z){
  g <- 1 / (1+exp(-z))
  return(g)
}

cost <- function(theta, X, y){
  m <- length(y)
  g <- sigmoid(X%*%theta)
  J <- (t(-y)%*%log(g)-t(1-y)%*%log(1-g)) / m
  return(J)
}

grad <- function(theta, X, y){
  m <- length(y)
  h <- sigmoid(X%*%theta)
  grad <- (t(X)%*%(h-y)) / m
  return(grad)
}

logistic_reg <- function(X, y){
  theta <- matrix(rep(0, ncol(X)), nrow=ncol(X))
  cost_opti <- optim(theta, fn=cost, gr=grad, X=X, y=y)
  return(cost_opti$par)
}

logistic_prob <- function(theta, X){
  return(sigmoid(X%*%theta))
}

logistic_pred <- function(prob){
  return(round(prob, 0))
}

theta <- logistic_reg(X, y)
prob <- logistic_prob(theta, X)
pred <- logistic_pred(prob)

#check whether our from-scratch model gives the same parameter with glm
logreg1 <- glm(admit ~., data=my_data, family="binomial")
summary(logreg1)$coefficients

##logistic regression with glm function
my_data2 <- read.csv("data.csv")
str(my_data2)
colSums(is.na(my_data2))
table(my_data2$target)

#drop unnecessary columns
my_data2 <- my_data2[, !names(my_data2) %in% c("X", "song_title", "artist")]

#randomly split data
split = sample.split(my_data2$target, SplitRatio=0.75)
train = subset(my_data2, split==TRUE)
test = subset(my_data2, split==FALSE)

#build the model on training set
logreg2 = glm(target ~ ., data=train, family="binomial")
summary(logreg2)

#prediction on training set
train_pred <- predict(logreg2, type="response")

#analyze predictions on training set
summary(train_pred)
tapply(train_pred, train$target, mean)
table(train$target, train_pred > 0.5)
table(train$target, train_pred > 0.7)
table(train$target, train_pred > 0.4)

#ROC
ROCRpred <- prediction(train_pred, train$target)
ROCRperf <- performance(ROCRpred, "tpr", "fpr")
plot(ROCRperf, colorize=TRUE)

#prediction on testing set
test_pred <- predict(logreg2, newdata=test, type="response")

#analyze predictions on testing set
summary(test_pred)
tapply(test_pred, test$target, mean)
table(test$target, test_pred > 0.5)

#find optimal cutoff
opt_cutoff <- optimalCutoff(test$target, test_pred)[1]

#check misclassification, AUROC, and concordance
misClassError(test$target, test_pred, threshold=opt_cutoff)
plotROC(test$target, test_pred)
Concordance(test$target, test_pred)