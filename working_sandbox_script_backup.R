library(reticulate)
# use_condaenv("baseclone")
use_python("/usr/bin/python3", required=TRUE)
py_config()
source_python(file="import_functions.py")
library(tensorflow)
#tensorflow::install_tensorflow()
library(keras)
library(dplyr)
library(ggplot2)
library(e1071)


path_to_data = "/home/taras/R_scripts/BigDataCup/"
file_name = "trainingSet.RData"
load(paste0(path_to_data, file_name))

#data = convert_json_data_to_csv(path_to_data,file_name)
#data %>% as_tibble() 
data$TIME = rep(1:60, dim(data)[1]/60)

# first timeseries plots
data %>% 
  mutate(ID = as.factor(ID), LABEL = as.factor(LABEL)) %>% 
  #  select(ABSNJZH, TIME, ID, LABEL) %>% 
  filter(ID %in% 1:100) %>% #group_by(ID) %>% 
  reshape2::melt(id.vars=c("ID", "LABEL", "TIME", "FOLD"), na.rm=TRUE) %>%
  ggplot(aes(x=TIME, y=value, colour=LABEL, group=ID)) + 
  geom_line(size=0.25) + 
  facet_wrap(variable~., scales="free_y", nrow=5)

ggsave("facet_grid_all_variables.png", width=10, height=10,units="in", dpi=600)

# mean/variance analysis
variable = "ABSNJZH"
summarisedTOT = data %>% 
  mutate(ID = as.factor(ID), LABEL = as.factor(LABEL), TIME=as.factor(TIME)) %>% 
  select(variable, TIME, ID, LABEL) %>% 
  group_by(ID, LABEL) %>% 
  summarise_if(is.numeric, list(~mean(., na.rm=TRUE), ~sd(., na.rm=TRUE), ~skewness(., na.rm=TRUE)), na.rm=TRUE) 
summarisedTOT %>% na.omit() %>% 
  #  filter(TOTBSQ > 0 & TOTBSQ < 5e10) %>% 
  #  ggplot(aes(x=mean, y=sd, colour=LABEL)) + geom_point(size=0.2) + 
  ggplot(aes(x=mean/sd, colour=LABEL)) + stat_ecdf() + 
  facet_wrap(LABEL~.)

library(rgl)
options(rgl.printRglwidget = TRUE)
#plot3d(summarisedTOT$mean, summarisedTOT$sd, summarisedTOT$skewness, col = 'red', type = 's', radius=.05)
plot3d(summarisedTOT$mean, summarisedTOT$sd, summarisedTOT$skewness, col = as.numeric(summarisedTOT$LABEL),  type = 's', size=.6)


# trasforming data for decision tree
filtered = summarisedTOT %>% na.omit() %>% ungroup() %>% select(LABEL, mean, sd, skewness)
sample_size = dim(filtered)[1]
train_size = round(sample_size*.8)  
test_size = sample_size - train_size
data_train = filtered[1:train_size,]
data_test  = filtered[(train_size+1):sample_size,]
#y_train = filtered[1:train_size, 1]
#y_test  = filtered[(train_size+1):sample_size, 1]

# decision tree
library(rpart)
library(rpart.plot)
library(MLmetrics)
fit <- rpart(LABEL~., data = data_train, method = 'class', model = T) 
# draw the decision tree 
rpart.plot(fit, type = 4, extra = 101)

predictions = as.numeric(rpart.predict(fit, data_test[,2:4], type="vector")-1)
sklearn.metrics = import("sklearn.metrics")
sklearn.metrics$f1_score(data_test$LABEL, as.integer(predictions), average='binary', labels="[0, 1]")

MLmetrics::Accuracy(predictions, data_test$LABEL)
MLmetrics::f1_score(data_test$LABEL, as.integer(predictions))


fit_logit = glm(LABEL ~ ., data=data_train, family=binomial(link="logit"))
predicted <- plogis(predict(fit_logit, data_test[,2:4]))  # predicted scores
predictions = ifelse(predicted > .5, 1, 0)

summarisedTOTsd = data %>% 
  mutate(ID = as.factor(ID), LABEL = as.factor(LABEL), TIME=as.factor(TIME)) %>% 
  select(TOTBSQ, TIME, ID, LABEL) %>% 
  group_by(ID, LABEL) %>% 
  summarise_if(is.numeric, e1071::skewness, na.rm=TRUE) 
summarisedTOTsd %>% 
  #  filter(TOTBSQ > 0 & TOTBSQ < 5e10) %>% 
  ggplot(aes(x=TOTBSQ, colour=LABEL)) + geom_histogram()


# dealing with NAs
number_of_NA = data %>% group_by(ID) %>% summarise_all(list(~sum(is.na(.))))  # count number of NAs per ID
IDs_with_most_NAs = number_of_NA[which(number_of_NA$EPSX > 6), "ID"]  # 6 is the maximum number of NAs per ID allowed!
data_clean = data %>% filter( !(ID %in% unlist(IDs_with_most_NAs)) )
#library(norm)
#s = data_clean %>% as.matrix() %>% prelim.norm()

# data standardization
data_st = data_clean
means = data_st %>% 
  mutate(ID = as.factor(ID), LABEL = as.factor(LABEL), TIME = as.factor(TIME), FOLD = as.factor(FOLD)) %>%
  summarise_if(is.numeric, mean, na.rm=TRUE)
#  summarise_all(mean, na.rm=TRUE)
SDs = data_st %>% 
  mutate(ID = as.factor(ID), LABEL = as.factor(LABEL), TIME = as.factor(TIME), FOLD = as.factor(FOLD)) %>%
  summarise_if(is.numeric, sd, na.rm=TRUE)
#  summarise_all(sd, na.rm=TRUE)

data_st[1:25] = scale(data_st[1:25], center=as.numeric(means), scale=as.numeric(SDs))
data_st[is.na(data_st)] = 0

#data_st %>% 
#  mutate(ID = as.factor(ID), LABEL = as.factor(LABEL), TIME=as.factor(TIME), FOLD = as.factor(FOLD)) %>%
#  summarise_if(is.numeric, mean, na.rm=TRUE)

#data_st %>% as_tibble()

# trasforming data for Keras
sample_size = dim(data_st)[1]/60
train_size = round(sample_size*.8)  
test_size = sample_size - train_size
x_array = aperm(array(t(data_st[1:25]), dim=c(25, 60, sample_size)), perm=c(3,1,2))
y_array = as.array(array(t(data_st[26]), dim=c(60, sample_size))[1,])
x_train = x_array[1:train_size,,]
x_test  = x_array[(train_size+1):sample_size,,]
y_train = y_array[1:train_size]
y_test  = y_array[(train_size+1):sample_size]

# callbacks
checkpoint_dir <- "checkpoints_conv2d"
dir.create(checkpoint_dir, showWarnings = FALSE)
filepath <- file.path(checkpoint_dir, "model.{epoch:02d}-{val_loss:.2f}.hdf5")

# Create checkpoint callback
cp_callback <- callback_model_checkpoint(
  filepath = filepath,
  save_weights_only = FALSE,
  verbose = 1
)



# Keras model - Try #1: Dense layers
x_train = array_reshape(x_train, c(nrow(x_train), 25*60))
x_test  = array_reshape(x_test, c(nrow(x_test), 25*60))
#y_train <- to_categorical(y_train, 2)
#y_test <- to_categorical(y_test, 2)
model1 <- keras_model_sequential() 
model1 %>% 
  layer_dense(units = 128, activation = 'relu', input_shape = c(1500)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units =  64, activation = 'relu') %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units =  16, activation = 'relu') %>% 
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 1, activation = 'sigmoid')


model1 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_rmsprop(lr=0.0001),
  metrics = c('accuracy')
)

history <- model1 %>% fit(
  x_train, y_train, 
  epochs = 10, batch_size = 128, 
  validation_split = 0.2
)

model1 %>% evaluate(x_test, y_test)

# check different aperm:
# 27006*60*25


# Keras model - Try #2: Conv1D layers
x_train = array_reshape(x_train, c(nrow(x_train), 25*60, 1))
x_test  = array_reshape(x_test, c(nrow(x_test), 25*60, 1))
#y_train <- to_categorical(y_train, 2)
#y_test <- to_categorical(y_test, 2)
model2 <- keras_model_sequential() 
model2 %>% 
  layer_conv_1d(kernel_size=5, filters=32, activation="relu", input_shape=list(25*60, 1)) %>% 
  layer_max_pooling_1d(pool_size=3) %>% 
  layer_conv_1d(filters=32, kernel_size=5, activation="relu") %>% 
  layer_gru(units=32, dropout=0.1, recurrent_dropout=0.5) %>% 
  layer_dense(units=1, activation="sigmoid")

model2 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_rmsprop(lr=0.0001),
  metrics = c('accuracy')
)

history <- model2 %>% fit(
  x_train, y_train, 
  epochs = 10, batch_size = 128, 
  validation_split = 0.2
)


# Keras model - Try #3: Conv2D layers 
x_train = array_reshape(x_train, c(nrow(x_train), 25, 60, 1))
x_test  = array_reshape(x_test, c(nrow(x_test), 25, 60, 1))
#y_train <- to_categorical(y_train, 2)
#y_test <- to_categorical(y_test, 2)
model3 <- keras_model_sequential() 
model3 %>% 
  layer_conv_2d(kernel_size=5, filters=32, activation="relu", input_shape=list(25, 60, 1)) %>% 
  layer_max_pooling_2d(pool_size=3) %>% 
  layer_conv_2d(filters=32, kernel_size=5, activation="relu") %>% 
  layer_reshape(target_shape=c(1344, 1)) %>% 
  layer_gru(units=32, dropout=0.1, recurrent_dropout=0.5) %>% 
  layer_dense(units=1, activation="sigmoid")

model3 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_rmsprop(lr=0.0001),
  metrics = c('accuracy')
)

history <- model3 %>% fit(
  x_train, y_train, 
  epochs = 10, batch_size = 64,
  verbose=2, shuffle = TRUE, 
  validation_split = 0.2,
  callbacks = list(cp_callback)  # pass callback to training
)



# Evaluate the predictions
sklearn.metrics = import("sklearn.metrics")
#scores = confusion_matrix(df_val_labels, pred_labels).ravel()
#tn, fp, fn, tp = scores
#print('TN:{}\tFP:{}\tFN:{}\tTP:{}'.format(tn, fp, fn, tp))
f1 = sklearn.metrics$f1_score(df_val_labels, pred_labels, average='binary', labels=[0, 1])
print('f1-score = {}'.format(f1))




# read test dataset:
file_name = "testSet.json"
#test_data = convert_test_json_data_to_csv(path_to_data,file_name)

file_name = "testSet.RData"
load(paste0(path_to_data, file_name))


#test_data[1:25] = scale(test_data[1:25], center=as.numeric(means), scale=as.numeric(SDs))
test_data[1:25] = scale(test_data[1:25], center=TRUE, scale=TRUE)

test_data[is.na(test_data)] = 0
x_testSet_array = aperm(array(t(test_data[1:25]), dim=c(25, 60, 173512)), perm=c(3,1,2))
x_test = array_reshape(x_testSet_array, c(nrow(x_testSet_array), 25*60, 1))

predictions = model1 %>% predict_classes(x_test)
results = data.frame(Id = 1:length(predictions), ClassLabel = predictions)
write.csv(results, file="submission6_logit_on_TOTBSQ.csv", row.names=FALSE)


# kaggle competitions submit bigdata2019-flare-prediction -f submission5_conv_net_1d_with_gru_seq.csv -m "My fifth submission: conv 1d with GRU - with dense"

