# Xtream gradiant boosting for classfication model

setwd("D:\\Machine")

#import packages and dataset
library(Matrix)
library(xgboost)
library(magrittr)
library(dplyr)

data <- read.csv("Binary.csv")
data$rank <- as.factor(data$rank)
# data$admit <- as.factor(data$admit)
str(data)

# data partition 
library(caret)
set.seed(1234)

# samp <- sample.int(nrow(data),size = 0.8*nrow(data))

sam <- createDataPartition(data$admit,times = 1,p=0.8,list = F)
train <- data[sam,]
test <- data[-sam,]

prop.table(table(data$admit))
prop.table(table(train$admit))
prop.table(table(test$admit))


# create matrix One-hot-encoding for factor variables
# it will create dummy variable for factor variables
trainm <- sparse.model.matrix(admit ~.-1,data = train)
head(trainm)
train_label <- train[,"admit"]
trainMatrix <- xgb.DMatrix(data = as.matrix(trainm),label = train_label)

testM <- sparse.model.matrix(admit ~.-1,data = test)
test_lablel <- test[,"admit"]
testMatrix <- xgb.DMatrix(data = as.matrix(testM),label = test_lablel)

#parameter (nc means no.of unique classes)
nc <- length(unique(train_label))
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = nc)
watchlist <- list(train=trainMatrix,test = testMatrix)

 # extreme Gradient Boosting model
bst_model <- xgb.train(params = xgb_params,
                       data = trainMatrix,
                       nrounds = 100,watchlist = watchlist)

# Training and testing error plot
bst_model
e <- data.frame(bst_model$evaluation_log)
plot(e$iter,e$train_mlogloss,col = 'blue')
lines(e$iter,e$test_mlogloss,col = 'red')

min(e$test_mlogloss)
e[e$test_mlogloss== 0.630034,]

# let's explore more eta default is 0.3 but it ranges from 0 to 1 
# if eta value is low more robust to overfitting
bst_model <- xgb.train(params = xgb_params,
                       data = trainMatrix,
                       nrounds = 100,watchlist = watchlist,eta =0.05)
e <- data.frame(bst_model$evaluation_log)
plot(e$iter,e$train_mlogloss,col='blue')
lines(e$iter,e$test_mlogloss,col='red')

min(e$test_mlogloss)
e[e$test_mlogloss== 0.599299,]

bst_model <- xgb.train(params = xgb_params,
                       data = trainMatrix,
                       nrounds = 29,watchlist = watchlist,eta =0.05)


# feature importance
imp <- xgb.importance(colnames(trainMatrix),model = bst_model)
imp #gain value is more important (gpa)
xgb.plot.importance(imp)

# prediction & confusion matrix
p <- predict(bst_model,newdata = testMatrix)
head(p)
# need get in proper order 0.73 and 0.26 is same 1
pred <- matrix(p,nrow = nc,ncol = length(p)/nc) %>% 
        t() %>% data.frame() %>% 
        mutate(label=test_lablel,max_prob=max.col(.,"last")-1)

# why -1 class or 2 insted 1 & 2 -1 is consider as a 0 & 1
head(pred)

table(prediction = pred$max_prob,Actual = pred$label)
57/80

bst_model <- xgb.train(params = xgb_params,
                       data = trainMatrix,
                       nrounds = 387,watchlist = watchlist,
                       eta = 0.001,max.depth = 3,gamma = 0,
                       subsample = 1,colsample_bytree = 1,
                       missing = NA,seed=333)


e <- data.frame(bst_model$evaluation_log)
plot(e$iter,e$train_mlogloss,col='blue')
lines(e$iter,e$test_mlogloss,col='red')

min(e$test_mlogloss)
e[e$test_mlogloss== 0.631221,]


# prediction & confusion matrix
p <- predict(bst_model,newdata = testMatrix)
head(p)
# need get in proper order 0.73 and 0.26 is same 1
pred <- matrix(p,nrow = nc,ncol = length(p)/nc) %>% 
  t() %>% data.frame() %>% 
  mutate(label=test_lablel,max_prob=max.col(.,"last")-1)

# why -1 class or 2 insted 1 & 2 -1 is consider as a 0 & 1
head(pred)

table(prediction = pred$max_prob,Actual = pred$label)

61/80






# Fit glm model: model
model <- glm(admit ~ ., family = "binomial", train)

# Predict on test: p

train$pred_y <- predict(model,train,type = 'response')
train$pred_gp <- ifelse(train$pred_y>=0.4,'Yes','No')

# confusion matrix
head(train$admit)
head(train$pred_y)

tab1 <- table(actual=train$admit,predicted=train$pred_gp)
tab1
sum(diag(tab1))/sum(tab1)
prop.table(tab1,1)###shows percentage




test$pred_y <- predict(model,test,type = 'response')
test$pred_gp <- ifelse(test$pred_y>=0.4,'Yes','No')

tab2 <- table(actual=test$admit,predicted=test$pred_gp)
tab2
sum(diag(tab2))/sum(tab2)
prop.table(tab2,1)###shows percentage


#Roc curve
# Predict on test: p
p <- predict(model, test, type = "response")

# Make ROC curve
library(pROC)
library(ROCR)

Rocrpred <- prediction(p,test$admit)
Rocrper <- performance(Rocrpred,"tpr","fpr")
plot(Rocrper,colorize=TRUE,print.cutoffs.at=seq(0.1,by=0.1))

library(pROC)
auc(train$admit,train$pred_y)
auc(test$admit,test$pred_y)


#Customizing trainControl

# Create trainControl object: myControl
myControl <- trainControl(
  method = "cv",
  number = 10,
  summaryFunction = twoClassSummary,
  classProbs = TRUE, # IMPORTANT!
  verboseIter = TRUE
)

# Don't forget the classProbs argument to train control, 
# especially if you're going to calculate AUC or logloss.

# Train glm with custom trainControl: model
model <- train(
  admit ~ ., 
  train, 
  method = "glm",
  trControl = myControl
)

# Print model to console
model

#Random forest
# Fit random forest: model
model <- train(
  admit~.,
  tuneLength = 1,
  data = data, 
  method = 'ranger',
  trControl = trainControl(
    method = "cv", 
    number = 5, 
    verboseIter = TRUE
  )
)

# Print model to console
model

# Fit random forest: model
model <- train(
  admit ~ .,
  tuneLength = 3,
  data = data, 
  method = "ranger",
  trControl = trainControl(
    method = "cv", 
    number = 5, 
    verboseIter = TRUE
  )
)

# print the model
model

# Plot model
plot(model)
# From previous step
tuneGrid <- data.frame(
  .mtry = c(2, 3, 5),
  .splitrule = "variance",
  .min.node.size = 5)


#Glmnet  
# Lasso regression and ridge regression
#Advantage of glmnet
#What's the advantage of glmnet over regular glm models?

# Create custom trainControl: myControl
myControl <- trainControl(
  method = "cv", 
  number = 10,
  summaryFunction = twoClassSummary,
  classProbs = TRUE, # IMPORTANT!
  verboseIter = TRUE
)

# Fit glmnet model: model
model <- train(
  admit ~ ., 
  data,
  method = "glmnet",
  trControl = myControl
)

# Print model to console
model

# Print maximum ROC statistic
max(model[["results"]][["ROC"]])

# Note: This glmnet will use AUC rather than accuracy to select the final model parameters.

#Why a custom tuning grid?
# Why use a custom tuning grid for a glmnet model?
  # The default tuning grid is very small and 
# there are many more potential glmnet models you want to explore.
# Train glmnet with custom trainControl and tuning: model

model <- train(
 admit ~ ., 
  data,
  tuneGrid = expand.grid(
    alpha = 0:1,
    lambda = seq(0.0001, 1, length = 20)
  ),
  method = "glmnet",
  trControl = myControl
)

# Print model to console
model

# Print maximum ROC statistic
max(model[["results"]][["ROC"]])

# Interpreting glmnet plots
# Here's the tuning plot for the custom tuned glmnet model 
# you created in the last exercise. For the overfit dataset, 
# which value of alpha is better?
# alpha = 0 (ridge)
# alpha = 1 (lasso) it is best 










