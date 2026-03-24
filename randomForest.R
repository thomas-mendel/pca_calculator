set.seed(SEED)

results_list <- list()

make_objective <- function(fold_index) {

  function(trials) {

    selected_idx <- trials$suggest_int("subset", 8, min(16, length(FS_randomForest_2)))
    feats <- FS_randomForest_2[1:selected_idx]

    train_df <- analysis(folds$splits[[fold_index]]) %>%
      dplyr::select(all_of(c("target", feats)))
    val_df   <- assessment(folds$splits[[fold_index]]) %>%
      dplyr::select(all_of(c("target", feats)))

    params <- list(
      ntree    = trials$suggest_int("ntree",    200, 5000),
      mtry     = trials$suggest_int("mtry",     2, ncol(train_df)-1),
      nodesize = trials$suggest_int("nodesize", 1, 5),
      maxnodes = trials$suggest_int("maxnodes", 10, 50),
      sampsize = trials$suggest_int("sampsize", 25, nrow(train_df)),
      importance = TRUE,
      probability = TRUE
    )

    mod <- randomForest::randomForest(
      x = train_df %>% dplyr::select(-target),
      y = train_df$target,
      ntree     = params$ntree,
      mtry      = params$mtry,
      nodesize  = params$nodesize,
      maxnodes  = params$maxnodes,
      sampsize  = params$sampsize,
      importance = params$importance,
      probability = params$probability
    )

    prob_train <- predict(mod, train_df, type = "prob")
    prob_val   <- predict(mod, val_df,   type = "prob")

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