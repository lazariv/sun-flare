library(reticulate)
use_condaenv("baseclone")  # for tensorflow-1.10
#use_condaenv("baseclone-tf14", required=TRUE) # for tensorflow-1.14
# use_python("/usr/bin/python3", required=TRUE)
py_config()
#source_python(file="import_functions.py")
library(tensorflow)
#tensorflow::install_tensorflow()
library(keras)
library(dplyr)
library(ggplot2)

path_to_data = "/lustre/ssd/lazariv/SunFlare/"
file_name = "trainingSet_scaled.RData"
load(paste0(path_to_data, file_name))

# define metrics
source(paste0(path_to_data, "metrics.R"))

model_name = "ConvNet1d_with_linear_V5_tuning"

FLAGS <- flags(
  flag_numeric("spat_dropout1", 0.2),
  flag_numeric("spat_dropout2", 0.2),
  flag_numeric("dropout1", 0.2),
  flag_numeric("dropout2", 0.2),
  flag_integer("filters1", 256),
  flag_integer("filters2",  32),
  flag_integer("denseunits1",128),
  flag_integer("denseunits2", 16),
  flag_string("optimizer", "adamax"),
  flag_numeric("lr", 0.002, "Learning rate for Adam optimizer")
)



#data_st %>% 
#  mutate(ID = as.factor(ID), LABEL = as.factor(LABEL), TIME=as.factor(TIME), FOLD = as.factor(FOLD)) %>%
#  summarise_if(is.numeric, mean, na.rm=TRUE)

#data_st %>% as_tibble()

# trasforming data for Keras
sample_size = dim(data_st)[1]/60
train_size = round(sample_size*.9)  
test_size = sample_size - train_size
x_array = aperm(array(t(data_st[1:26]), dim=c(26, 60, sample_size)), perm=c(3,2,1))
y_array = as.array(array(t(data_st["LABEL"]), dim=c(60, sample_size))[1,])
train_rows = sample(sample_size, size=train_size) 

x_train = x_array[train_rows,,]
x_test  = x_array[-train_rows,,]
y_train = y_array[train_rows]
y_test  = y_array[-train_rows]



# callbacks
checkpoint_dir <- paste0("CP_", model_name)
dir.create(checkpoint_dir, showWarnings = FALSE)
filepath <- file.path(checkpoint_dir, "model.{epoch:02d}-{val_loss:.4f}-{val_f1:.4f}.hdf5")

# Create checkpoint callback
cp_callback <- callback_model_checkpoint(
  filepath = filepath, save_weights_only = FALSE, verbose = 1, period = 5
)


# Keras model - Try #2: Conv1D layers
#x_train = array_reshape(x_train, c(nrow(x_train), 25*60, 1))
#x_test  = array_reshape(x_test, c(nrow(x_test), 25*60, 1))
model1 <- keras_model_sequential() 
model1 %>% 
  layer_conv_1d(kernel_size=5, filters=FLAGS$filters1, activation="relu", padding="same", input_shape=list(60, 26), 
                kernel_regularizer=regularizer_l1_l2(l1 = 0.01, l2 = 0.01)) %>% 
  layer_spatial_dropout_1d(FLAGS$spat_dropout1) %>%
  layer_max_pooling_1d(pool_size=3) %>% 
  layer_conv_1d(filters=FLAGS$filters2, kernel_size=5, activation="relu", kernel_regularizer=regularizer_l1_l2(l1 = 0.01, l2 = 0.01)) %>% 
  layer_spatial_dropout_1d(FLAGS$spat_dropout2) %>%
  layer_flatten() %>%
  layer_dense(units=FLAGS$denseunits1, activation="relu") %>%
  layer_dropout(FLAGS$dropout1) %>%
  layer_dense(units=FLAGS$denseunits2, activation="relu") %>%
  layer_dropout(FLAGS$dropout2) %>%
  layer_dense(units=1, activation="sigmoid")

model1 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_adamax(lr=FLAGS$lr, decay=0.0001),
#  optimizer = FLAGS$optimizer,
  metrics = c('accuracy', custom_metric("f1", f1_m))
#  metrics = c('accuracy')
)

history <- model1 %>% fit(
  x_train, y_train, 
  epochs = 3, batch_size = 128,
  verbose=2, shuffle = TRUE, 
  validation_split = 0.2,
  callbacks = list(#cp_callback, 
                   callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.5, verbose = 1, cooldown = 5)
#          #         ,callback_early_stopping(monitor = "val_loss", verbose=1, mode="min", patience=5, restore_best_weights=TRUE)
                   )  # pass callback to training
)

png(type="cairo", width = 1800, height = 1800)
plot(history)
dev.off()
model1 %>% save_model_hdf5(paste0("my_", model_name, ".h5"))
# load with:
# model1 = load_model_hdf5("my_ConvNet1d_with_linear_V5_tuning_with_06476.h5", custom_objects=list("f1"=f1_m))

#save.image("backup.RData")

model1 %>% evaluate(x_test, y_test, verbose = 2)
predictions = model1 %>% predict_classes(x_test)
# predicted = model1 %>% predict_proba(x_test)
# optCutOff = InformationValue::optimalCutoff(actuals=as.numeric(y_test), predictedScores=predicted, returnDiagnostics=F) 
# predictions = ifelse(predicted > optCutOff, 1, 0)

# calculate metrics on test dataset
cat("Accuracy =", MLmetrics::Accuracy(predictions, as.numeric(y_test)), "\n")
# Accuracy = 0.8746192
cat("F1_Score =", try(MLmetrics::F1_Score(y_test, as.integer(predictions), positive="1")), "\n")
# F1_Score = 0.9234831


file_name = "testSet.RData"
load(paste0(path_to_data, file_name))


#test_data[1:25] = scale(test_data[1:25], center=as.numeric(means), scale=as.numeric(SDs))
test_data[1:25] = scale(test_data[1:25], center=TRUE, scale=TRUE)

test_data[is.na(test_data)] = 0

test_data$LINEAR = rep(seq(-1, 1, length.out=60), dim(test_data)[1]/60)
test_data = test_data %>% select(LINEAR, everything())

x_testSet_array = aperm(array(t(test_data[1:26]), dim=c(26, 60, 173512)), perm=c(3,2,1))
x_test = x_testSet_array
#x_test = array_reshape(x_testSet_array, c(nrow(x_testSet_array), 25*60, 1))


predictions = model1 %>% predict_classes(x_test)
results = data.frame(Id = 1:length(predictions), ClassLabel = predictions)
readr::write_csv(x=results, path=paste0("submissions/submission26_", model_name, ".csv") )
#write.csv(results, file="submission5_conv_net_1d_with_gru_seq.csv", row.names=FALSE)

# kaggle competitions submit bigdata2019-flare-prediction -f submissions/submission22_ConvNet1d_with_linear_V4_tuning.csv -m "My 22th submission: ConvNet1D with linear variable with adamax"
