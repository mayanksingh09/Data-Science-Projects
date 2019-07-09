#setwd("/Users/..")
source("Cleaning_Code.R")

library(dplyr)

## Creating index for rows
nbatrain$Index <- 1:length(nbatrain$Team)
nbatest$Index <- 1:length(nbatest$Team)


## Subsetting dataset to use only required variables

colnames(nbatrain)

## Simple linear model
nbatrain_subset <- nbatrain[c(3, 41:51)]
nbatrain_subset$Home <- ifelse(nbatrain_subset$Home == "Home", 1, 0)

nbatest_subset <- nbatest[c(3, 41:51)]
nbatest_subset$Home <- ifelse(nbatest_subset$Home == "Home", 1, 0)

fit.lm <- lm(pointdiff_perc ~., data = nbatrain_subset[, -which(names(nbatrain_subset) == "Index")])
summary(fit.lm)

predlm_test <- predict(fit.lm, newdata = nbatest_subset)

summary(predlm_test)
summary(nbatest_subset$pointdiff_perc)

SSE <- sum((predlm_test - nbatest_subset$pointdiff_perc)^2)
SST <- sum((mean(nbatrain_subset$pointdiff_perc) - nbatest_subset$pointdiff_perc)^2)

## Out of sample R2 value
1 - SSE/SST


## Random Forest
library(randomForest)

fit.rf <- randomForest(pointdiff_perc ~., data = nbatrain_subset[, -which(names(nbatrain_subset) %in% c("Index", "Opponent"))])

predrf_test <- predict(fit.rf, newdata = nbatest_subset)

SSE.rf <- sum((predrf_test - nbatest_subset$pointdiff_perc)^2)
SST.rf <- sum((mean(nbatrain_subset$pointdiff_perc) - nbatest_subset$pointdiff_perc)^2)

1 - SSE.rf/SST.rf

varImpPlot(fit.rf)

## SVM

library(e1071)
library(rpart)

fit.svm <- svm(pointdiff_perc~., 
                  data = nbatrain_subset[, -which(names(nbatrain_subset) == "Index")], 
                  type="eps-regression", 
                  kernel="linear", cost = 10, gamma = 1)

summary(fit.svm)

predsvm_test <- predict(fit.svm, newdata = nbatest_subset)
#summary(predrf_test)
#summary(predsvm_test)

SSE.svm <- sum((predsvm_test - nbatest_subset$pointdiff_perc)^2)
SST.svm <- sum((mean(nbatrain_subset$pointdiff_perc) - nbatest_subset$pointdiff_perc)^2)

1 - SSE.rf/SST.rf

## Caret
library(caret)
control <- trainControl(method="repeatedcv", number=5, repeats=2, search="random")

set.seed(123)
mtry <- sqrt(ncol(nbatrain_subset))
rf_random <- train(pointdiff_perc~., 
                   data = nbatrain_subset[, -which(names(nbatrain_subset) == "Index")], 
                   method="rf", 
                   metric="rmse", 
                   tuneLength=15, 
                   trControl=control)
print(rf_random)
plot(rf_random)


set.seed(123)
tunegrid <- expand.grid(.mtry=c(1:15))
rf_gridsearch <- train(pointdiff_perc~.,
                       data=nbatrain_subset[, -which(names(nbatrain_subset) == "Index")],
                       method="rf",
                       metric="rmse",
                       tuneGrid=tunegrid, 
                       trControl=control)
print(rf_gridsearch)
plot(rf_gridsearch)


## Neural Net

library(keras)
#install_keras()


x.holdout <- model.matrix(pointdiff_perc ~., 
                         data = nbatest_subset[, -which(names(nbatest_subset) == "Index")])[,-1]
y.holdout <- nbatest_subset$pointdiff_perc

x.data <- model.matrix(pointdiff_perc ~., data=nbatrain_subset[, -which(names(nbatrain_subset) == "Index")])[,-1]
y.data <- nbatrain_subset$pointdiff_perc

#rescale (to be between 0 and 1)
x_train <- x.data %*% diag(1/apply(x.data, 2, function(x) max(x, na.rm = TRUE)))
y_train <- as.numeric(y.data)
x_test <- x.holdout %*% diag(1/apply(x.data, 2, function(x) max(x, na.rm = TRUE)))
y_test <- as.numeric(y.holdout) 

#rescale (unit variance and zero mean)
mean <- apply(x.data,2,mean)
std <- apply(x.data,2,sd)
x_train <- scale(x.data,center = mean, scale = std)
y_train <- as.numeric(y.data)
x_test <- scale(x.holdout,center = mean, scale = std)
y_test <- as.numeric(y.holdout) 

num.inputs <- ncol(x_test)


model <- keras_model_sequential() %>%
        layer_dense(units=16,activation="relu",input_shape = c(num.inputs)) %>%
        layer_dense(units=16,activation="relu") %>%
        layer_dense(units=16,activation="relu") %>%
        layer_dense(units=1,activation="sigmoid")

summary(model)

model %>% compile(
        loss = 'mean_squared_error',
        optimizer = 'adam'
)

history <- model %>% fit(
        x_train, y_train, 
        epochs = 30, batch_size = 128, 
        validation_split = 0.25
)
results.NN1 <- model %>% evaluate(x_train,y_train)
results.NN1

results.NN1 <- model %>% evaluate(x_test,y_test)
results.NN1

pred.NN1 <- model%>% predict(x_test)

# PerformanceMeasure <- function(actual, prediction, threshold=.5) {
#         1-mean( abs( (prediction>threshold) - actual ) )  
#         #R2(y=actual, pred=prediction, family="binomial")
#         #1-mean( abs( (prediction- actual) ) )  
# }
# 
# PerformanceMeasure(actual=y_test, prediction=pred.NN1, threshold=.5)

SSE.nn <- sum((as.vector(pred.NN1) - nbatest_subset$pointdiff_perc)^2)
SST.nn <- sum((mean(nbatrain_subset$pointdiff_perc) - nbatest_subset$pointdiff_perc)^2)

1 - SSE.nn/SST.nn





############# Using MLR ################

library(mlr)

trainTask <- makeRegrTask(data = nbatrain_subset, target = "pointdiff_perc")
testTask <- makeRegrTask(data = nbatest_subset, target = "pointdiff_perc")

# Normalize the variables
trainTask <- normalizeFeatures(trainTask, method = "standardize")
testTask <- normalizeFeatures(testTask, method = "standardize")

# Drop features
trainTask <- dropFeatures(task = trainTask, features = c("Index"))
testTask <- dropFeatures(task = testTask, features = c("Index"))

## Linear Regression (tuned)

linear.learner <- makeLearner("regr.lm")
cv.linear <- crossval(learner = linear.learner, 
                      task = trainTask, 
                      iters = 15,
                      measures = rmse, 
                      show.info = T)


fmodel <- train(linear.learner, trainTask)
getLearnerModel(fmodel)

fpmodel <- predict(fmodel, testTask)

SSE.lm.tune <- sum((fpmodel$data$truth - fpmodel$data$response)^2)
SST.lm.tune <- sum((mean(nbatrain_subset$pointdiff_perc) - nbatest_subset$pointdiff_perc)^2)

1 - SSE.lm.tune/SST.lm.tune

nbatest$pred_lm <- fpmodel$data$response

## CART (tuned)

makeatree <- makeLearner("regr.rpart")
set_cv <- makeResampleDesc("CV",
                           iters = 15)
gs <- makeParamSet(
        makeIntegerParam("minsplit", lower = 10, upper = 50),
        makeIntegerParam("minbucket", lower = 5, upper = 50),
        makeNumericParam("cp", lower = 0.001, upper = 0.2)
)
gscontrol <- makeTuneControlGrid()
stune <- tuneParams(learner = makeatree, 
                    resampling = set_cv, 
                    task = trainTask, 
                    par.set = gs, 
                    control = gscontrol, 
                    measures = rmse)

stune$x
stune$y

t.tree <- setHyperPars(makeatree, par.vals = stune$x)
t.rpart <- train(t.tree, trainTask)
getLearnerModel(t.rpart)

tpmodel <- predict(t.rpart, testTask)

SSE.cart.tune <- sum((tpmodel$data$truth - tpmodel$data$response)^2)
SST.cart.tune <- sum((mean(nbatrain_subset$pointdiff_perc) - nbatest_subset$pointdiff_perc)^2)

1 - SSE.cart.tune/SST.cart.tune

nbatest$pred_cart <- tpmodel$data$response

library(rattle)
plot(t.rpart$learner.model)
text(t.rpart$learner.model,use.n=TRUE,cex=.5)


## Random forest (tuned)

getParamSet("regr.randomForest")

rf <- makeLearner("regr.randomForest",
                  par.vals = list(ntree = 200, 
                                  mtry = 3))

rf$par.vals <- list(importance = TRUE)

#set tunable parameters
#grid search to find hyperparameters
rf_param <- makeParamSet(
        makeIntegerParam("ntree",lower = 50, upper = 500),
        makeIntegerParam("mtry", lower = 3, upper = 10),
        makeIntegerParam("nodesize", lower = 10, upper = 50)
)

rancontrol <- makeTuneControlRandom(maxit = 50L)

# Set 5 fold CV
set_cv <- makeResampleDesc("CV",iters = 5L)

rf_tune <- tuneParams(learner = rf, 
                      resampling = set_cv, 
                      task = trainTask, 
                      par.set = rf_param, 
                      control = rancontrol, 
                      measures = rmse)


#cv rmse
rf_tune$y

# best parameters
rf_tune$x

#using hyperparameters for modeling
rf.tree <- setHyperPars(rf, par.vals = rf_tune$x)

rforest <- train(rf.tree, trainTask)

rfmodel <- predict(rforest, testTask)
SSE.rf.tune <- sum((rfmodel$data$truth - rfmodel$data$response)^2)
SST.rf.tune <- sum((mean(nbatrain_subset$pointdiff_perc) - nbatest_subset$pointdiff_perc)^2)

1 - SSE.rf.tune/SST.rf.tune

nbatest$pred_rf <- rfmodel$data$response

# Get feature importance for variables
featureimp_rf <- getFeatureImportance(rforest)
featureimp_rf <- as.data.frame(featureimp_rf$res)
featureimp_rf <- data.frame(t(featureimp_rf))
featureimp_rf <- tibble::rownames_to_column(featureimp_rf)
colnames(featureimp_rf) <- c("Variable", "Var.Imp")

featureimp_rf <- featureimp_rf %>% 
        arrange(desc(Var.Imp))

library(ggplot2)
ggplot(data = featureimp_rf, aes(x = reorder(Variable, Var.Imp), y = Var.Imp)) + geom_bar(stat = "identity") + xlab("Variables") + ylab("Variable Importance") + ggtitle("Variable Importance Plot") + theme(plot.title = element_text(hjust = 0.5)) + coord_flip()

## SVM tuned

getParamSet("regr.ksvm") #do install kernlab package 
ksvm <- makeLearner("regr.ksvm")


pssvm <- makeParamSet(
        makeDiscreteParam("C", values = 2^c(-8,-4,-2,0)), #cost parameters
        makeDiscreteParam("sigma", values = 2^c(-8,-4,0,4)) #RBF Kernel Parameter
)

# Search function
ctrl <- makeTuneControlGrid()

# tune model
res <- tuneParams(ksvm,
                  task = trainTask,
                  resampling = set_cv,
                  par.set = pssvm,
                  control = ctrl,
                  measures = rmse)

# CV RMSE
res$y

# best parameters for the model
t.svm <- setHyperPars(ksvm, par.vals = res$x)

# train the model
par.svm <- train(ksvm, trainTask)

# predict
predict.svm <- predict(par.svm, testTask)

SSE.svm.tune <- sum((predict.svm$data$truth - predict.svm$data$response)^2)
SST.svm.tune <- sum((mean(nbatrain_subset$pointdiff_perc) - nbatest_subset$pointdiff_perc)^2)

1 - SSE.svm.tune/SST.svm.tune

featureimp_svm <- mlr::generateFeatureImportanceData(trainTask, 
                                   "permutation.importance", 
                                   ksvm, nmc = 3)
featureimp_svm <- as.data.frame(featureimp_svm$res)
featureimp_svm <- data.frame(t(featureimp_svm))
featureimp_svm <- tibble::rownames_to_column(featureimp_svm)
colnames(featureimp_svm) <- c("Variable", "Var.Imp")

featureimp_svm <- featureimp_svm %>% 
        arrange(desc(Var.Imp))


ggplot(data = featureimp_svm, aes(x = reorder(Variable, Var.Imp), y = Var.Imp)) + geom_bar(stat = "identity") + xlab("Variables") + ylab("Variable Importance") + ggtitle("Variable Importance Plot") + theme(plot.title = element_text(hjust = 0.5)) + coord_flip()


## GBM

getParamSet("regr.gbm")
g.gbm <- makeLearner("regr.gbm")

# tuning method
rancontrol <- makeTuneControlRandom(maxit = 50L)

# 3 fold CV
set_cv <- makeResampleDesc("CV",iters = 3L)

# parameters
gbm_par<- makeParamSet(
#        makeDiscreteParam("distribution", values = "bernoulli"),
        makeIntegerParam("n.trees", lower = 100, upper = 1000), #number of trees
        makeIntegerParam("interaction.depth", lower = 2, upper = 10), #depth of tree
        makeIntegerParam("n.minobsinnode", lower = 10, upper = 80), # min observations in a tree node
        makeNumericParam("shrinkage",lower = 0.01, upper = 1) # how fast the algorithm learns/moves
)

# tune parameters
tune_gbm <- tuneParams(learner = g.gbm, 
                       task = trainTask,
                       resampling = set_cv,
                       measures = rmse,
                       par.set = gbm_par,
                       control = rancontrol)

# tuned cv
tune_gbm$y

# set final parameters
final_gbm <- setHyperPars(learner = g.gbm, par.vals = tune_gbm$x)

# train final model
to.gbm <- train(final_gbm, trainTask)

# make predictions
pr.gbm <- predict(to.gbm, testTask)

SSE.gbm.tune <- sum((pr.gbm$data$truth - pr.gbm$data$response)^2)
SST.gbm.tune <- sum((mean(nbatrain_subset$pointdiff_perc) - nbatest_subset$pointdiff_perc)^2)

1 - SSE.gbm.tune/SST.gbm.tune

nbatest$pred_gbm <- pr.gbm$data$response

## XGBOOST

# load xgboost
set.seed(123)
getParamSet("regr.xgboost")


# make learner with inital parameters
xg_set <- makeLearner("regr.xgboost")
xg_set$par.vals <- list(
        objective = "reg:linear",
        eval_metric = "rmse",
        nrounds = 250
)


# define parameters for tuning
xg_ps <- makeParamSet(
        makeIntegerParam("nrounds",lower=200,upper=600),
        makeIntegerParam("max_depth",lower=3,upper=20),
        makeNumericParam("lambda",lower=0.30,upper=0.60),
        makeNumericParam("eta", lower = 0.001, upper = 0.5),
        makeNumericParam("subsample", lower = 0.10, upper = 0.80),
        makeNumericParam("min_child_weight",lower=1,upper=5),
        makeNumericParam("colsample_bytree",lower = 0.2,upper = 0.8)
)

# define search function
rancontrol <- makeTuneControlRandom(maxit = 100L) #do 100 iterations

# 3 fold cross validation
set_cv <- makeResampleDesc("CV",iters = 3L)

# tune parameters
xg_tune <- tuneParams(learner = xg_set, 
                      task = trainTask, 
                      resampling = set_cv,
                      measures = rmse,
                      par.set = xg_ps, 
                      control = rancontrol)

# set parameters
xg_new <- setHyperPars(learner = xg_set, par.vals = xg_tune$x)

# train model
xgmodel <- train(xg_new, trainTask)

# test model
predict.xg <- predict(xgmodel, testTask)

SSE.xg.tune <- sum((predict.xg$data$truth - predict.xg$data$response)^2)
SST.xg.tune <- sum((mean(nbatrain_subset$pointdiff_perc) - nbatest_subset$pointdiff_perc)^2)

1 - SSE.xg.tune/SST.xg.tune

nbatest$pred_xgb <- predict.xg$data$response

#write.csv(nbatest, file = "nbatest_withpredictions.csv", row.names = F)


################# EXTRA #################

#selecting top 6 important features (worse)
top_task <- filterFeatures(trainTask, method = "randomForest.importance", abs = 6)
top_test <- filterFeatures(testTask, method = "randomForest.importance", abs = 6)

xgmodel2 <- train(xg_new, top_task)
predict.xg2 <- predict(xgmodel2, top_test)

SSE.xg.tune2 <- sum((predict.xg2$data$truth - predict.xg2$data$response)^2)
SST.xg.tune2 <- sum((mean(nbatrain_subset$pointdiff_perc) - nbatest_subset$pointdiff_perc)^2)

1 - SSE.xg.tune2/SST.xg.tune2
