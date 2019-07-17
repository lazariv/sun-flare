##########################################################################
### Script for loading data and fitting gradient boosting algorithms   ###
### VERSION 2: the "TOTBSQ" and "TOTUSJH" variables are used           ###
##########################################################################

# libraries
library(reticulate)
# use_condaenv("baseclone")
use_python("/usr/bin/python3", required=TRUE)
py_config()
library(dplyr)
library(ggplot2)
library(e1071)

### data 
path_to_data = "/home/taras/taras_personal/BigDataCup/"
file_name = "trainingSet.RData"
load(paste0(path_to_data, file_name))

# mean/variance analysis
variables = c("ABSNJZH", "R_VALUE", "SAVNCPP", "TOTBSQ","TOTUSJH")
#variables = c("TOTUSJH", "TOTBSQ")
#variables = c("EPSX", "MEANALP")

summarisedTOT = data %>% 
  mutate(ID = as.factor(ID), LABEL = as.factor(LABEL), TIME=as.factor(TIME)) %>% 
  select(variables, TIME, ID, LABEL) %>% 
  group_by(ID, LABEL) %>% 
  summarise_if(is.numeric, list(~mean(., na.rm=TRUE), ~sd(., na.rm=TRUE), ~skewness(., na.rm=TRUE)), na.rm=TRUE) 
  #summarise_if(is.numeric, list(~mean(., na.rm=TRUE), ~sd(., na.rm=TRUE), ~quantile(., probs=0.25, na.rm=TRUE), ~quantile(., probs=c(0.75), na.rm=TRUE)), na.rm=TRUE) 

#summarisedTOT %>% na.omit() %>% 
#  #  filter(TOTBSQ > 0 & TOTBSQ < 5e10) %>% 
#  ggplot(aes(x=sd, y=skewness, colour=LABEL)) + geom_point(size=0.2)


# trasforming data for log regression
filtered = summarisedTOT %>% na.omit() %>% ungroup() %>% select(-ID) 
#filtered[-1] = scale(filtered[-1], center=TRUE, scale=TRUE)
sample_size = dim(filtered)[1]
train_size = round(sample_size*.8)  
test_size = sample_size - train_size
train_rows = sample(sample_size, size=train_size) 
#data_train = filtered[1:train_size,]
#data_test  = filtered[(train_size+1):sample_size,]
data_train = filtered[train_rows,]
data_test  = filtered[-train_rows,]
#y_train = filtered[1:train_size, 1]
#y_test  = filtered[(train_size+1):sample_size, 1]



library(xgboost)
fit_xgboost = xgboost(data = as.matrix(data_train[,-1]), label = as.numeric(data_train$LABEL)-1, max_depth = 5, eta = 1,
                      nrounds = 5, objective = "binary:logistic")
summary(fit_xgboost)
xgb.dump(fit_xgboost, 'model.dump')
predicted = plogis(predict(fit_xgboost, as.matrix(data_test[,-1])))
optCutOff <- InformationValue::optimalCutoff(data_test$LABEL, predicted)
predictions = ifelse(predicted > optCutOff, 1, 0)  # using a cutoff = 0.5
predictions %>% table

# calculate metrics on test dataset
cat("Accuracy =", MLmetrics::Accuracy(predictions, data_test$LABEL))
# Accuracy = 0.8859593
cat("F1_Score =", MLmetrics::F1_Score(data_test$LABEL, as.integer(predictions)))
# F1_Score = 0.9324417



library(caret)
#library(doParallel)
#cl <- makeForkCluster()
#registerDoParallel(cl)
#stopCluster(cl)

library(doMC)
registerDoMC(10)
f1 <- function(data, lev = NULL, model = NULL) {
#  print(data)
  f1_val <- MLmetrics::F1_Score(y_pred = data$pred, y_true = data$obs, positive="1")
  c(F1 = f1_val)
}
#mat = lapply(c("LogitBoost", 'xgbTree', 'rf', 'svmRadial'),
#             function (met) {
#               train(LABEL ~ ., method=met, data=filtered)
#             })

xgb_grid <- expand.grid(
  nrounds = c(10, 100, 200),
  eta = c(0.3, 0.1),
  max_depth = c(3,5,7),
  gamma = 0,
  colsample_bytree=c(.8, 1.0), 
  min_child_weight=1,
  subsample=c(0.8, 1.0)
)
fit_xgboost = train(LABEL ~ ., method="xgbTree", data=filtered, metric="F1", 
               trControl = trainControl(number = 5,
                                        summaryFunction=f1),
               tuneGrid = xgb_grid)

predictions = predict(fit_rf, data_test[,-1])
predictions %>% table
MLmetrics::F1_Score(y_true=data_test$LABEL, y_pred=predictions, positive="1")




