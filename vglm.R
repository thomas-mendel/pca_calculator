set.seed(SEED)

results_list <- list()

make_objective <- function(fold_index) {

  function(trials) {

    # --------------------------------------------------
    # Feature selection subset
    # --------------------------------------------------
    selected_idx <- trials$suggest_int("subset", 8, min(16, length(FS_vglm_2)))
    feats <- FS_vglm_2[1:selected_idx]

    train_df <- analysis(folds$splits[[fold_index]]) %>%
      dplyr::select(all_of(c("target", feats)))
    val_df   <- assessment(folds$splits[[fold_index]]) %>%
      dplyr::select(all_of(c("target", feats)))
    test_df  <- test_df_15 %>%
      dplyr::select(all_of(c("target", feats)))


    params <- list(
      family_type = trials$suggest_categorical("family_type", c("cratio", "cumulative", "acat", "propodds")),
      maxit   = trials$suggest_int("maxit", 10L, 100L),
      epsilon = trials$suggest_loguniform("epsilon", 1e-4, 1),
      half_stepsizing = trials$suggest_categorical("half_stepsizing",c(TRUE, FALSE)),
      stepsize = trials$suggest_float("stepsize", 0.5, 1.0),
      xarg = trials$suggest_categorical("xarg", c(TRUE, FALSE)),
      yarg = trials$suggest_categorical("yarg", c(TRUE, FALSE)),
      model_flag = trials$suggest_categorical("model", c(FALSE, TRUE)),
      smart = trials$suggest_categorical("smart", c(TRUE, FALSE)),
      parallel = trials$suggest_categorical("parallel", c(TRUE, FALSE))
    )
    family_obj <- switch(
      params$family_type,
      "cratio"     = VGAM::cratio(parallel = params$parallel),
      "acat"       = VGAM::acat(parallel = TRUE),
      "cumulative" = VGAM::cumulative(parallel = TRUE),
      "propodds"   = VGAM::propodds()
    )
    ctrl <- VGAM::vglm.control(
      maxit           = params$maxit,
      epsilon         = params$epsilon,
      half.stepsizing = params$half_stepsizing,
      stepsize        = params$stepsize,
      trace           = FALSE
    )
    mod <- VGAM::vglm(
      formula = target ~ .,
      family  = family_obj,
      data    = train_df,
      control = ctrl,
      method  = "vglm.fit",   # IRLS
      model   = params$model_flag,
      x.arg   = params$xarg,
      y.arg   = params$yarg,
      smart  = params$smart
    )

    prob_train <- predict(mod, newdata = train_df, type = "response")
    prob_val   <- predict(mod, newdata = val_df,   type = "response")
    prob_test  <- predict(mod, newdata = test_df,  type = "response")

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

for (k in 1:3) {
  cat("Running fold:", k, "\n")
  sampler <- optuna$samplers$TPESampler(seed = 0L)
  study <- optuna$create_study(directions = c("maximize", "maximize"), sampler = sampler)
  study$optimize(make_objective(k), n_trials = 300)
}

results_df <- as.data.frame(do.call(rbind, lapply(results_list, as.data.frame)))