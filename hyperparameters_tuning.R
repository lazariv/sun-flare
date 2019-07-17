Sys.setenv("R_DEFAULT_DEVICE"="pdf")
options(bitmapType = 'cairo')
options(device = 'pdf')

library(reticulate)
use_condaenv("baseclone")
# use_python("/usr/bin/python3", required=TRUE)
py_config()
#source_python(file="import_functions.py")
library(tensorflow)
#tensorflow::install_tensorflow()
library(keras)
library(dplyr)
library(ggplot2)
library(tfruns)


runs_dir = "model_tuning"

tuning_run("ConvNet_V4_tuning.R", runs_dir = runs_dir, flags = list(
  dropout1 = c(0.3, 0.5),
  dropout2 = c(0.4),
  filters1 = c(128, 512),
  filters2 = c( 32,  64),
  denseunits1 = c(32, 128),
  denseunits2 = c( 4, 16),
#  optimizer = c("sgd", "rmsprop", "adagrad", "adadelta", "adam", "adamax", "nadam"),
  lr = c(0.0002, 0.00002)
))


# list runs witin the specified runs_dir
ls_runs(order = eval_acc, runs_dir = runs_dir)
