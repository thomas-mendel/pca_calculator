set.seed(SEED)

results_list <- list()

make_objective <- function(fold_index) {

  function(trials) {

    selected_idx <- trials$suggest_int("subset", 8, min(16, length(FS_rpart_2)))
    feats <- FS_rpart_2[1:selected_idx]

    params <- list(
        cp = trials$suggest_loguniform("cp", 1e-3, 0.05),               
        minsplit = trials$suggest_int("minsplit", 25, 50),
        minbucket = trials$suggest_int("minbucket", 8, 16),                
        maxdepth = trials$suggest_int("maxdepth", 2, 5),                 
        xval = 10,            
        split = "information"
    )

    train_df <- analysis(folds$splits[[fold_index]]) %>%
      dplyr::select(all_of(c("target", feats)))
    val_df <- assessment(folds$splits[[fold_index]]) %>%
      dplyr::select(all_of(c("target", feats)))

    mod <- rpart::rpart(
              formula = target ~ .,
              data    = train_df,
              method  = "class",
              parms   = list(split = params$split),
              control = rpart::rpart.control(
                          cp            = params$cp,
                          minsplit      = params$minsplit,
                          minbucket     = params$minbucket,
                          maxdepth      = params$maxdepth,
                          xval          = params$xval
                        )
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