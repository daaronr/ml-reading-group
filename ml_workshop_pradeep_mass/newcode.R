rm(list=ls())

library(MASS)
library(leaps)
library(glmnet)
library(tree)
library(randomForest)
library(gbm)

# attaching the data
#attach(Boston)

# Data set description
?Boston
dim(Boston)
summary(Boston)

# converting chas to a qualitative variable
Boston$chas <- as.factor(chas)

# Training and Test data
set.seed(123)
train <- sample(1:nrow(Boston), nrow(Boston)/2)

Boston.train <- Boston[train,]
Boston.test <- Boston[-train,]

# Least Squares Model

lm.fit <- lm(medv ~ ., data=Boston.train)
summary(lm.fit)

# Out of sample performance
lm.pred <- predict(lm.fit,newdata = Boston.test)
lm.test.mse <- mean((lm.pred - Boston.test$medv)^2) 
sqrt(lm.test.mse) # Residual standard error

tss <- sum((mean(Boston.test$medv) - Boston.test$medv)^2) 
lm.rss <- sum((lm.pred - Boston.test$medv)^2) 
lm.test.r2 <- 1 - lm.rss/tss

# Forward Stepwise Selection
set.seed(123)
reg.fwd <- regsubsets(medv ~ ., data = Boston.train, nvmax = 13, method = "forward")
reg.summary.fwd <- summary(reg.fwd)
reg.summary.fwd

# Choosing between models of different sizes
par(mfrow = c(2, 2))
# AIC or Cp
plot(reg.summary.fwd$cp, xlab = "No. of variables", ylab = "Cp/AIC", type = "l")
points(which.min(reg.summary.fwd$cp), reg.summary.fwd$cp[which.min(reg.summary.fwd$cp)], col = "red", cex = 2, pch = 20)

# BIC  
plot(reg.summary.fwd$bic, xlab = "No. of variables", ylab = "BIC", type = "l")
points(which.min(reg.summary.fwd$bic), reg.summary.fwd$bic[which.min(reg.summary.fwd$bic)], col = "red", cex = 2, pch = 20)

# Adj - R2
plot(reg.summary.fwd$adjr2, xlab = "No. of variables", ylab = "Adj. R2", type = "l")
points(which.max(reg.summary.fwd$adjr2), reg.summary.fwd$adjr2[which.max(reg.summary.fwd$adjr2)], col = "red", cex = 2, pch = 20)

# Using test set
test.mat <- model.matrix(medv~., data=Boston.test) # X variables of the test data
fwd.test.error=rep(NA,13)
fwd.test.rss=rep(NA,13)
for (i in 1:13){
  coefi=coef(reg.fwd,id=i)
  pred=test.mat[,names(coefi)] %*% coefi
  fwd.test.error[i]=mean((Boston.test$medv - pred)^2)
  fwd.test.rss[i]=sum((Boston.test$medv - pred)^2)
}

plot(fwd.test.error, xlab = "No. of variables", ylab = "Test MSE", type = "l")
points(which.min(fwd.test.error), fwd.test.error[which.min(fwd.test.error)], col = "red", cex = 2, pch = 20)

fwd.test.mse <- fwd.test.error[which.min(fwd.test.error)]
fwd.test.r2 <- 1 - fwd.test.rss[which.min(fwd.test.error)]/tss

# LASSO model
train.X <- model.matrix(medv ~ ., data = Boston.train)
test.X <- model.matrix(medv ~ ., data = Boston.test)
train.Y <- Boston.train[,'medv']
test.Y <- Boston.test[,'medv']
set.seed(123)
lasso.fit <- glmnet(train.X, train.Y, alpha = 1)
par(mfrow = c(1, 1))
plot(lasso.fit,label = TRUE)

# Tuning the LASSO model - cross-validation
set.seed(123)
cv.out <- cv.glmnet(train.X, train.Y, alpha = 1)
plot(cv.out)

# Model with least CV error
bestlam1 <- cv.out$lambda.min
bestlam1

# Most shrunken model within 1 std. error of the min CV error
bestlam2 <- cv.out$lambda.1se
bestlam2

# coefficients of both models
predict(lasso.fit,type="coefficients",s=bestlam1)
predict(lasso.fit,type="coefficients",s=bestlam2)

lasso.pred1 <- predict(lasso.fit, s = bestlam1, newx = test.X)
lasso.pred2 <- predict(lasso.fit, s = bestlam2, newx = test.X)

# Test MSE
lasso1.mse <- mean((lasso.pred1 - test.Y)^2) 
lasso2.mse <- mean((lasso.pred2 - test.Y)^2) 

# Test R2
lasso1.rss <- sum((lasso.pred1 - test.Y)^2) 
lasso2.rss <- sum((lasso.pred2 - test.Y)^2) 

lasso1.test.r2 <- 1 - lasso1.rss/tss
lasso2.test.r2 <- 1 - lasso2.rss/tss

# Ridge model
set.seed(123)
ridge.fit <- glmnet(train.X, train.Y, alpha = 0)
plot(ridge.fit,label = TRUE)

# Tuning the model - cross-validation
set.seed(123)
cv.out.ridge <- cv.glmnet(train.X, train.Y, alpha = 0)
plot(cv.out.ridge)

# Model with least CV error
bestlam.ridge <- cv.out.ridge$lambda.min
bestlam.ridge

predict(ridge.fit,type="coefficients",s=bestlam.ridge)

ridge.pred <- predict(ridge.fit, s = bestlam.ridge, newx = test.X)

# Test MSE
ridge.mse <- mean((ridge.pred - test.Y)^2) 

# Test R2
ridge.rss <- sum((ridge.pred - test.Y)^2) 
ridge.test.r2 <- 1 - ridge.rss/tss


##### Regression Tree model #####
set.seed(123)
regtree.fit  <- tree(medv ~ ., data = Boston.train)
summary(regtree.fit )
plot(regtree.fit ) # to plot tree structure
text(regtree.fit , pretty = 0) # to plot the node labels

regtree.pred  <- predict(regtree.fit, newdata = Boston.test)
regtree.mse <- mean((regtree.pred - Boston.test$medv)^2)
# Test R2
regtree.rss <- sum((regtree.pred - Boston.test$medv)^2) 
regtree.test.r2 <- 1 - regtree.rss/tss

# Tuning (Pruning) the tree - to find the optimal level of complexity
set.seed(123)
cv.fit <- cv.tree(regtree.fit) 
cv.fit 
par(mfrow=c(1,2))
plot(cv.fit$size, cv.fit$dev, type = "b") 

# tree- pruning  
prune.fit <- prune.tree(regtree.fit,best = 8)
plot(prune.fit)
text(prune.fit,pretty=0)

### Bagging and Random Forest models ####
set.seed(123)
bagging.fit <- randomForest(medv ~ ., data = Boston.train, mtry = 13, importance = TRUE)
bagging.fit
bagging.pred <- predict(bagging.fit, newdata = Boston.test)
bagging.mse <- mean((bagging.pred  - test.Y)^2)  
# Test R2
bagging.rss <- sum((bagging.pred  - test.Y)^2)  
bagging.test.r2 <- 1 - bagging.rss/tss

# Varying the number of variables to be considered at each node
set.seed(123)
rf.test.error=rep(NA,13)
rf.test.rss=rep(NA,13)
for (i in 1:13){
  rf.fit <- randomForest(medv ~ ., data = Boston.train, mtry = i, importance = TRUE)
  pred=predict(rf.fit, newdata = Boston.test)
  rf.test.error[i]=mean((Boston.test$medv - pred)^2)
  rf.test.rss[i]=sum((Boston.test$medv - pred)^2)
}
rf.test.error
which.min(rf.test.error)
rf.test.mse <- rf.test.error[which.min(rf.test.error)]
rf.test.r2 <- 1 - rf.test.rss[which.min(rf.test.error)]/tss

rf5.fit <- randomForest(medv ~ ., data = Boston.train, mtry = 5, importance = TRUE)

rf5.fit$importance
#varImpPlot(rf5.fit)

## Boosting model
set.seed(123)
boost.fit <- gbm(medv~.,data = Boston.train, distribution = "gaussian",n.trees = 5000,interaction.depth = 4, shrinkage = 0.01)
summary(boost.fit)

# partial dependence plots
par(mfrow=c(1,2))
plot(boost.fit,i="rm")
plot(boost.fit,i="lstat")

# Tuning the boosting model
s  <- 10^seq(-1, -3, length = 10)
boost.test.error=matrix(data=NA,nrow=5,ncol=10)
boost.test.rss=matrix(data=NA,nrow=5,ncol=10)
set.seed(123)
for (i in 1:5) {
  for (j in 1:10){
boosting <- gbm(medv~.,data = Boston.train, distribution = "gaussian",n.trees = 5000,interaction.depth = i, shrinkage = s[j])
pred <- predict(boosting, newdata=Boston.test,n.trees = 5000)    
boost.test.error[i,j]=mean((Boston.test$medv - pred)^2)
boost.test.rss[i,j]=sum((Boston.test$medv - pred)^2)    
  }
} 

# Test error and out of sample R2
boost.test.mse <- min(boost.test.error)
loc <- which(boost.test.error == boost.test.mse,arr.ind = TRUE)
boost.test.r2 <- 1 - boost.test.rss[loc]/tss

# Creating Table
test.mse <- c(lm.test.mse,fwd.test.mse,lasso1.mse,ridge.mse,regtree.mse,bagging.mse,rf.test.mse,boost.test.mse)
test.r2 <- c(lm.test.r2,fwd.test.r2,lasso1.test.r2,ridge.test.r2,regtree.test.r2,bagging.test.r2,rf.test.r2,boost.test.r2)
outofsample <- matrix(c(test.mse,test.r2),ncol = 2)
rownames(outofsample) <- c("OLS", "Forward Stepwise" ,"Lasso", "Ridge", "Regression Tree", "Bagging", "Random Forest", "Boosting")
colnames(outofsample) <- c("Test MSE", "Test R-sqr")
outofsample





