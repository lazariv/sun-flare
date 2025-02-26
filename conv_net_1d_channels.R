library(reticulate)
use_condaenv("baseclone")
# use_python("/usr/bin/python3", required=TRUE)
py_config()
source_python(file="import_functions.py")
library(tensorflow)
#tensorflow::install_tensorflow()
library(keras)
library(dplyr)
library(ggplot2)


path_to_data = "/lustre/ssd/lazariv/SunFlare/"
file_name = "trainingSet_cleaned.RData"
load(paste0(path_to_data, file_name))


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
x_array = aperm(array(t(data_st[1:25]), dim=c(25, 60, sample_size)), perm=c(3,2,1))
y_array = as.array(array(t(data_st[26]), dim=c(60, sample_size))[1,])
x_train = x_array[1:train_size,,]
x_test  = x_array[(train_size+1):sample_size,,]
y_train = y_array[1:train_size]
y_test  = y_array[(train_size+1):sample_size]



# callbacks
checkpoint_dir <- "CP_conv1d_with_25_channels"
dir.create(checkpoint_dir, showWarnings = FALSE)
filepath <- file.path(checkpoint_dir, "model.{epoch:02d}-{val_loss:.2f}.hdf5")

# Create checkpoint callback
cp_callback <- callback_model_checkpoint(
  filepath = filepath,
  save_weights_only = FALSE,
  verbose = 1
)


# Keras model - Try #2: Conv1D layers
#x_train = array_reshape(x_train, c(nrow(x_train), 25*60, 1))
#x_test  = array_reshape(x_test, c(nrow(x_test), 25*60, 1))

model1 <- keras_model_sequential() 
model1 %>% 
  layer_conv_1d(kernel_size=5, filters=64, activation="relu", padding="same", input_shape=list(60, 25)) %>% 
  layer_max_pooling_1d(pool_size=3) %>% 
  layer_conv_1d(filters=32, kernel_size=5, activation="relu") %>% 
  layer_flatten() %>%
  layer_dense(units=16, activation="relu") %>%
  layer_dense(units=1, activation="sigmoid")

model1 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_adam(lr=0.0001),
  metrics = c('accuracy')
)

history <- model1 %>% fit(
  x_train, y_train, 
  epochs = 5, batch_size = 64,
  verbose=1, shuffle = TRUE, 
  validation_split = 0.2,
  callbacks = list(cp_callback)  # pass callback to training
)


model1 %>% save_model_hdf5("my_conv1d_with_25_channels.h5")

model1 %>% evaluate(x_test, y_test)
predictions = model1 %>% predict_classes(x_test)


# calculate metrics on test dataset
cat("Accuracy =", MLmetrics::Accuracy(predictions, as.numeric(y_test)))
# Accuracy = 0.8746192
cat("F1_Score =", MLmetrics::F1_Score(y_test, as.integer(predictions)))
# F1_Score = 0.9234831


file_name = "testSet.RData"
load(paste0(path_to_data, file_name))


#test_data[1:25] = scale(test_data[1:25], center=as.numeric(means), scale=as.numeric(SDs))
test_data[1:25] = scale(test_data[1:25], center=TRUE, scale=TRUE)

test_data[is.na(test_data)] = 0
x_testSet_array = aperm(array(t(test_data[1:25]), dim=c(25, 60, 173512)), perm=c(3,2,1))
x_test = x_testSet_array
#x_test = array_reshape(x_testSet_array, c(nrow(x_testSet_array), 25*60, 1))


predictions = model1 %>% predict_classes(x_test)
results = data.frame(Id = 1:length(predictions), ClassLabel = predictions)
readr::write_csv(x=results, path="submissions/submission15_ConvNet1D_with_channels.csv")

# kaggle competitions submit bigdata2019-flare-prediction -f submissions/submission15_ConvNet1D_with_channels.csv -m "My 15th submission: a ConvNet1D with channels as input"

