set.seed(SEED)

results_list <- list()

make_objective <- function(fold_index) {

  function(trials) {
    
    selected_idx <- trials$suggest_int("subset", 8, min(16, length(FS_multinom_2)))
    feats <- FS_multinom_2[1:selected_idx]

    train_df <- analysis(folds$splits[[fold_index]]) %>%
      dplyr::select(all_of(c("target", feats)))
    val_df <- assessment(folds$splits[[fold_index]]) %>%
      dplyr::select(all_of(c("target", feats)))

    y_train <- class.ind(train_df$target)
    y_val   <- class.ind(val_df$target)

    params <- list(
      size  = trials$suggest_int("size", 1L, 2L),
      decay = trials$suggest_float("decay", 0.01, 5),    
      maxit = trials$suggest_int("maxit", 1000L, 3000L),  
      rang  = trials$suggest_float("rang", 0.001, 0.1), 
      skip  = trials$suggest_categorical("skip", c(TRUE,FALSE)),
      MaxNWts   = trials$suggest_int("MaxNWts", 10000L, 100000L),
      abstol    = trials$suggest_float("abstol", 1e-10, 1e-5),
      reltol  = trials$suggest_float("reltol", 1e-10, 1e-5),
      softmax = TRUE,
      trace   = FALSE,
      Hess    = FALSE
    )


    mod <- nnet::nnet(
      x = as.matrix(train_df[, feats]),
      y = y_train,
      size      = params$size,
      decay     = params$decay,
      maxit     = params$maxit,
      rang      = params$rang,
      skip      = params$skip,
      softmax   = params$softmax,
      abstol    = params$abstol,
      reltol    = params$reltol,
      MaxNWts   = params$MaxNWts,
      trace     = params$trace
    )


    prob_train <- predict(mod, as.matrix(train_df[, feats]))
    prob_val   <- predict(mod, as.matrix(val_df[, feats]))

    auc_train <- calc_macro_auc(train_df$target, prob_train)
    auc_val   <- calc_macro_auc(val_df$target,   prob_val)
    acc_train <- calc_acc(train_df$target, prob_train)
    acc_val   <- calc_acc(val_df$target,   prob_val)

    results_list[[length(results_list) + 1]] <<- c(
      fold      = fold_index,
      params,
      subset    = paste(1:selected_idx, collapse = ","),
      auc_train = auc_train,
      auc_val   = auc_val,
      acc_train = acc_train,
      acc_val   = acc_val
    )


    return(c(acc_val, auc_val))
  }
}

# ============================================================
# Optuna tuning
# ============================================================
for (k in 1:3) {
  cat("Running fold:", k, "\n")
  sampler <- optuna$samplers$TPESampler(seed = 0L)
  study <- optuna$create_study(directions = c("maximize", "maximize"), sampler = sampler)
  study$optimize(make_objective(k), n_trials = 300)
}

results_df <- as.data.frame(do.call(rbind, lapply(results_list, as.data.frame)))