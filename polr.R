set.seed(SEED)

results_list <- list()

make_objective <- function(fold_index) {

  function(trials) {
    
    selected_idx <- trials$suggest_int("subset", 8, min(16, length(FS_polr_2)))
    feats <- FS_polr_2[1:selected_idx]

    train_df <- analysis(folds$splits[[fold_index]]) %>% 
      dplyr::select(all_of(c("target", feats)))
    val_df <- assessment(folds$splits[[fold_index]]) %>% 
      dplyr::select(all_of(c("target", feats)))
    test_df <- test_df_15 %>% 
      dplyr::select(all_of(c("target", feats)))

    params <- list(
      method = trials$suggest_categorical(
        "method", 
        c("logistic", "probit")
      ),
      maxit = trials$suggest_int("maxit", 100L, 5000L),
      reltol = trials$suggest_float("reltol", 1e-50, 1),
      Hess = TRUE,
      model = FALSE,
      trace = FALSE
    )

    mod <- MASS::polr(
      target ~ .,
      data = train_df,
      method = params$method,
      Hess = params$Hess,
      model = params$model,
      control = list(
        maxit = params$maxit,
        reltol = params$reltol,
        trace = params$trace
      )
    )

    prob_train <- predict(mod, train_df, type = "prob")
    prob_val   <- predict(mod, val_df,   type = "prob")
    prob_test  <- predict(mod, test_df,  type = "prob")

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