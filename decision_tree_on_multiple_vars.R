####################################################################
### Script for loading data and fitting a decision tree to data  ###
### VERSION 2: the "TOTBSQ" and "TOTUSJH" variables are used     ###
####################################################################
#

library(dplyr)
library(ggplot2)
library(e1071)
library(rpart)
library(rpart.plot)
library(MLmetrics)


path_to_data = "/home/taras/R_scripts/BigDataCup/"
file_name = "trainingSet.RData"
load(paste0(path_to_data, file_name))

# mean/variance analysis
variables = c("TOTBSQ","TOTUSJH")
summarisedTOT = data %>% 
  mutate(ID = as.factor(ID), LABEL = as.factor(LABEL), TIME=as.factor(TIME)) %>% 
  select(variables, TIME, ID, LABEL) %>% 
  group_by(ID, LABEL) %>% 
  summarise_if(is.numeric, list(~mean(., na.rm=TRUE), ~sd(., na.rm=TRUE), ~skewness(., na.rm=TRUE)), na.rm=TRUE) 
#summarisedTOT %>% na.omit() %>% 
#  #  filter(TOTBSQ > 0 & TOTBSQ < 5e10) %>% 
#  ggplot(aes(x=sd, y=skewness, colour=LABEL)) + geom_point(size=0.2)


# trasforming data for decision tree
filtered = summarisedTOT %>% na.omit() %>% ungroup() %>% select(-ID) #select(LABEL, mean, sd, skewness)
sample_size = dim(filtered)[1]
train_size = round(sample_size*.8)  
test_size = sample_size - train_size
data_train = filtered[1:train_size,]
data_test  = filtered[(train_size+1):sample_size,]
#y_train = filtered[1:train_size, 1]
#y_test  = filtered[(train_size+1):sample_size, 1]





# decision tree
fit <- rpart(LABEL~., data = data_train, method = 'class', model = T) 
# draw the decision tree 
rpart.plot(fit, type = 4, extra = 101)

predictions = as.numeric(rpart.predict(fit, data_test[,-1], type="vector")-1)

# calculate metrics on test dataset
cat("Accuracy =", MLmetrics::Accuracy(predictions, data_test$LABEL))
# Accuracy = 0.8746192
cat("F1_Score =", MLmetrics::F1_Score(data_test$LABEL, as.integer(predictions)))
# F1_Score = 0.9234831



#########################
### read test dataset ###
#########################

file_name = "testSet.RData"
load(paste0(path_to_data, file_name))

test_summarisedTOT = test_data %>%
  mutate(ID = as.factor(ID)) %>%
  select(variables, ID) %>%
  group_by(ID) %>%
  summarise_if(is.numeric, list(~mean(., na.rm=TRUE), ~sd(., na.rm=TRUE), ~skewness(., na.rm=TRUE)), na.rm=TRUE)

# check for NAs
#sum(is.na(test_summarisedTOT$TOTUSJH_mean))
#sum(is.na(test_summarisedTOT$TOTUSJH_sd))
#sum(is.na(test_summarisedTOT$TOTUSJH_skewness))
#test_summarisedTOT$TOTBSQ_skewness[is.na(test_summarisedTOT$TOTBSQ_skewness)] = 0
test_summarisedTOT[is.na(test_summarisedTOT)] = 0

# make predictions
predictions = as.numeric(rpart.predict(fit, test_summarisedTOT[,-1], type="vector")-1)
predictions %>% table

# write submission file
results = data.frame(Id = 1:length(predictions), ClassLabel = predictions)
readr::write_csv(x=results, path="submissions/submission9_a_decision_tree_on_2_vars.csv")

# kaggle competitions submit bigdata2019-flare-prediction -f submissions/submission9_a_decision_tree_on_2_vars.csv -m "My 9th submission: a decision tree on TOTBSQ and TOTUSJH"




