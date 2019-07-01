library(reticulate)
use_python("/usr/bin/python3", required=TRUE)
py_config()
source_python(file="import_functions.py")

path_to_data = "/home/taras/R_scripts/BigDataCup/"
file_name = "fold1Training.json"
data = convert_json_data_to_csv(path_to_data,file_name)
data$FOLD = 1

file_name = "fold2Training.json"
data = rbind(data, data.frame(convert_json_data_to_csv(path_to_data,file_name), FOLD = 2))

file_name = "fold3Training.json"
data = rbind(data, data.frame(convert_json_data_to_csv(path_to_data,file_name), FOLD = 3))

data$TIME = rep(1:60, 76773 + 92481 + 27006)
## Fold 1: 76773
## Fold 2: 92481
## Fold 3: 27006
data$ID[(76773*60+1):((76773+92481+27006)*60)] = data$ID[(76773*60+1):((76773+92481+27006)*60)] + 76773
data$ID[((76773+92481)*60+1):((76773+92481+27006)*60)] = data$ID[((76773+92481)*60+1):((76773+92481+27006)*60)] + 92481

save(data, file="trainingSet.RData")


# read test dataset:
file_name = "testSet.json"
test_data = convert_test_json_data_to_csv(path_to_data,file_name)

save(test_data, file="testSet.RData")
