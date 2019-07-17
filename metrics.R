
recall_m = function(y_true, y_pred) { 
  true_positives = k_sum(k_round(k_clip(y_true * y_pred, 0, 1)))
  possible_positives = k_sum(k_round(k_clip(y_true, 0, 1)))
  recall = true_positives / (possible_positives + k_epsilon())
  return(recall)
}
    
precision_m = function(y_true, y_pred) {
  true_positives = k_sum(k_round(k_clip(y_true * y_pred, 0, 1)))
  predicted_positives = k_sum(k_round(k_clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + k_epsilon())
  return(precision)
}
    
f1_m = function(y_true, y_pred) {
  precision = precision_m(y_true, y_pred)
  recall = recall_m(y_true, y_pred)
  return(2*((precision*recall)/(precision+recall+k_epsilon())) )
}  





# recall_m = function(y_true, y_pred) { 
#   true_positives = k_sum(k_round(k_clip(k_dot(y_true, y_pred), 0, 1)))
#   possible_positives = k_sum(k_round(k_clip(y_true, 0, 1)))
#   recall = true_positives / (possible_positives + k_epsilon())
#   return(recall)
# }
# 
# precision_m = function(y_true, y_pred) {
#   true_positives = k_sum(k_round(k_clip(k_dot(y_true, y_pred), 0, 1)))
#   predicted_positives = k_sum(k_round(k_clip(y_pred, 0, 1)))
#   precision = true_positives / (predicted_positives + k_epsilon())
#   return(precision)
# }
# 
# 
# 
# f1_metric <- custom_metric("f1", f1_m)
# 
# f1_m = function(y_true, y_pred) {
#   y_true = k_eval(y_true)
#   print(y_true)
#   y_pred = k_eval(y_pred)
#   print(y_pred)
#   f1_score = MLmetrics::F1_Score(y_true=y_true, y_pred=y_pred, positive="1")
#   print(f1_score)
#   return(k_constant(f1_score))
# }  
# 
