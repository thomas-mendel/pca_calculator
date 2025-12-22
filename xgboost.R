set.seed(SEED)

results_list <- list()

make_objective <- function(fold_index) {

  function(trials) {

    selected_idx <- trials$suggest_int("subset", 8, min(16, length(FS_xgboost_2)))
    feats <- FS_xgboost_2[1:selected_idx]

    train_df <- analysis(folds$splits[[fold_index]]) %>% dplyr::select(all_of(c("target", feats)))
    val_df   <- assessment(folds$splits[[fold_index]]) %>% dplyr::select(all_of(c("target", feats)))
    test_df  <- test_df_15 %>% dplyr::select(all_of(c("target", feats)))

    params <- list(
        objective = "multi:softprob",
        nthread = 5,
        eval_metric = "mlogloss",
        num_class = length(levels(train_df$target)),
        eta               = trials$suggest_float("eta", 0.01, 1, log=TRUE),
        max_depth         = trials$suggest_int("max_depth", 1, 4),
        min_child_weight  = trials$suggest_float("min_child_weight", 1, 2),
        lambda            = trials$suggest_float("lambda", 1, 15),
        alpha             = trials$suggest_float("alpha", 0.01, 1),
        grow_policy      = trials$suggest_categorical("grow_policy", c("depthwise", "lossguide")), 
        tree_method       = trials$suggest_categorical("tree_method", c("exact", "approx", "hist")),   
        nrounds           = trials$suggest_int("nrounds", 50, 200),
        gamma             = trials$suggest_float("gamma", 0.001, 5)
    )


    y_train <- as.numeric(train_df$target) - 1
    y_val   <- as.numeric(val_df$target) - 1
    y_test  <- as.numeric(test_df$target) - 1

    dtrain <- xgb.DMatrix(data = as.matrix(train_df[, feats]), label = y_train)
    dval   <- xgb.DMatrix(data = as.matrix(val_df[, feats]),   label = y_val)
    dtest  <- xgb.DMatrix(data = as.matrix(test_df[, feats]),  label = y_test)

    mod <- xgboost::xgb.train(
          params = list(
            objective = params$objective,
            nthread = params$nthread,
            eval_metric = params$eval_metric,
            num_class = params$num_class,
            eta              = params$eta,              
            max_depth        = params$max_depth,        
            min_child_weight = params$min_child_weight,
            lambda           = params$lambda,         
            alpha            = params$alpha,          
            grow_policy      = params$grow_policy,   
            tree_method      = params$tree_method,  
            gamma           =params$gamma
          ),
          data = dtrain,
          nrounds = params$nrounds,     
          verbose = FALSE
        )



    prob_train <- matrix(predict(mod, dtrain),
                         ncol = length(levels(train_df$target)),
                         byrow = TRUE)
    prob_val <- matrix(predict(mod, dval),
                       ncol = length(levels(train_df$target)),
                       byrow = TRUE)
    prob_test <- matrix(predict(mod, dtest),
                        ncol = length(levels(train_df$target)),
                        byrow = TRUE)
    colnames(prob_train) <- levels(train_df$target)
    colnames(prob_val)   <- levels(train_df$target)
    colnames(prob_test)  <- levels(train_df$target)


    auc_train <- calc_macro_auc(train_df$target, prob_train)
    auc_val   <- calc_macro_auc(val_df$target,   prob_val)
    auc_test  <- calc_macro_auc(test_df$target,  prob_test)
    acc_train <- calc_acc(train_df$target, prob_train)
    acc_val   <- calc_acc(val_df$target,   prob_val)
    acc_test  <- calc_acc(test_df$target,  prob_test)

    results_list[[length(results_list) + 1]] <<- c(
      fold      = fold_index,
      params,
      subset    = paste(1:selected_idx, collapse = ","),
      auc_train = auc_train,
      auc_val   = auc_val,
      auc_test  = auc_test,
      acc_train = acc_train,
      acc_val   = acc_val,
      acc_test  = acc_test
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