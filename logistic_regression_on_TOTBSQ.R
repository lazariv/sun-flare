##########################################################################
### Script for loading data and fitting a logistic regression to data  ###
### By default the "TOTBSQ" variable is used                           ###
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
path_to_data = "/home/taras/R_scripts/BigDataCup/"
file_name = "trainingSet.RData"
load(paste0(path_to_data, file_name))

# mean/variance analysis
summarisedTOT = data %>% 
  mutate(ID = as.factor(ID), LABEL = as.factor(LABEL), TIME=as.factor(TIME)) %>% 
  select(TOTBSQ, TIME, ID, LABEL) %>% 
  group_by(ID, LABEL) %>% 
  summarise_if(is.numeric, list(~mean(., na.rm=TRUE), ~sd(., na.rm=TRUE), ~skewness(., na.rm=TRUE)), na.rm=TRUE) 
summarisedTOT %>% na.omit() %>% 
  #  filter(TOTBSQ > 0 & TOTBSQ < 5e10) %>% 
  ggplot(aes(x=sd, y=skewness, colour=LABEL)) + geom_point(size=0.2)


# trasforming data for log regression
filtered = summarisedTOT %>% na.omit() %>% ungroup() %>% select(LABEL, mean, sd, skewness)
sample_size = dim(filtered)[1]
train_size = round(sample_size*.8)  
test_size = sample_size - train_size
data_train = filtered[1:train_size,]
data_test  = filtered[(train_size+1):sample_size,]
#y_train = filtered[1:train_size, 1]
#y_test  = filtered[(train_size+1):sample_size, 1]

# fitting regression with GLM
fit_logit = glm(LABEL ~ ., data=data_train, family=binomial(link="logit"))
predicted <- plogis(predict(fit_logit, data_test[,2:4]))  # predicted scores
predictions = ifelse(predicted > .5, 1, 0)  # using a cutoff = 0.5

# calculate metrics on test dataset
cat("Accuracy =", MLmetrics::Accuracy(predictions, data_test$LABEL))
# Accuracy = 0.8859593
cat("F1_Score =", MLmetrics::F1_Score(data_test$LABEL, as.integer(predictions)))
# F1_Score = 0.9324417

#########################
### read test dataset ###
#########################

file_name = "testSet.RData"
load(paste0(path_to_data, file_name))

test_summarisedTOT = test_data %>%
  mutate(ID = as.factor(ID)) %>%
  select(TOTBSQ, ID) %>%
  group_by(ID) %>%
  summarise_if(is.numeric, list(~mean(., na.rm=TRUE), ~sd(., na.rm=TRUE), ~skewness(., na.rm=TRUE)), na.rm=TRUE)

# check for NAs
sum(is.na(test_summarisedTOT$mean))
sum(is.na(test_summarisedTOT$sd))
sum(is.na(test_summarisedTOT$skewness))
test_summarisedTOT$skewness[is.na(test_summarisedTOT$skewness)] = 0


# make predictions
predicted <- plogis(predict(fit_logit, test_summarisedTOT[,2:4]))  # predicted scores
predictions = ifelse(predicted > .5, 1, 0)  # using a cutoff = 0.5
predictions %>% table



# write submission file
results = data.frame(Id = 1:length(predictions), ClassLabel = predictions)
readr::write_csv(x=results, path="submission6_logit_on_TOTBSQ.csv")



