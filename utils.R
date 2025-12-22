calc_acc <- function(true, prob_matrix) {
  pred_class <- colnames(prob_matrix)[max.col(prob_matrix)]
  mean(as.character(true) == pred_class)
}

calc_macro_auc <- function(true, prob_matrix) {
  levels_true <- levels(true)
  aucs <- sapply(seq_along(levels_true), function(i) {
    cls <- levels_true[i]
    y_bin <- as.numeric(true == cls)
    p_bin <- prob_matrix[, i]
    if (length(unique(y_bin)) == 1) return(NA_real_)
    pROC::roc(y_bin, p_bin, quiet = TRUE)$auc
  })
  mean(aucs, na.rm = TRUE)
}