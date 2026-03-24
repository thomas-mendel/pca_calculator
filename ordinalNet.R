set.seed(SEED)

results_list <- list()

make_objective <- function(fold_index) {

  function(trials) {

    selected_idx <- trials$suggest_int("subset", 8, min(16, length(FS_ordinalNet_2)))
    feats <- FS_ordinalNet_2[1:selected_idx]

    x_train <- analysis(folds$splits[[fold_index]]) %>% dplyr::select(all_of(c(feats))) %>% as.matrix()
    x_val   <- assessment(folds$splits[[fold_index]]) %>% dplyr::select(all_of(c(feats))) %>% as.matrix()
    y_train <- analysis(folds$splits[[fold_index]])$target
    y_val   <- assessment(folds$splits[[fold_index]])$target


    params <- list(
      alpha = trials$suggest_float("alpha", 0.05, 0.5),
      family = trials$suggest_categorical("family", c("cumulative", "sratio", "cratio", "acat")),
      link = trials$suggest_categorical("link", c("logit", "probit", "cloglog", "cauchit")),
      parallelTerms = trials$suggest_categorical("parallelTerms", c(TRUE, FALSE)),
      nonparallelTerms = trials$suggest_categorical("nonparallelTerms", c(TRUE, FALSE)), 
      nLambda = trials$suggest_int("nLambda", 20, 50),
      lambdaMinRatio = trials$suggest_float("lambdaMinRatio", 0.001, 0.05),
      maxiterOut = trials$suggest_int("maxiterOut", 50, 200),
      maxiterIn  = trials$suggest_int("maxiterIn", 50, 200),
      pMin = trials$suggest_float("pMin", 1e-8, 1e-4),
      threshOut = trials$suggest_float("threshOut", 1e-8, 1e-4)
    )

    if (!params$parallelTerms & !params$nonparallelTerms) {
      params$parallelTerms <- TRUE
    }


    mod <- ordinalNet::ordinalNet(
      x = x_train,
      y = y_train,
      alpha = params$alpha,
      family = params$family,
      link = params$link,
      parallelTerms = params$parallelTerms,
      nonparallelTerms = params$nonparallelTerms,
      nLambda = params$nLambda,
      lambdaMinRatio = params$lambdaMinRatio,
      threshOut = params$threshOut,
      threshIn = params$threshOut,
      pMin = params$pMin,
      maxiterOut = params$maxiterOut,
      maxiterIn = params$maxiterIn
    )


    prob_train <- predict(mod, x_train, type = "response")
    prob_val   <- predict(mod, x_val,   type = "response")

    colnames(prob_train) <- target_levels
    colnames(prob_val)  <- target_levels


    auc_train <- calc_macro_auc(y_train, prob_train)
    auc_val   <- calc_macro_auc(y_val,   prob_val)

    acc_train <- calc_acc(y_train, prob_train)
    acc_val   <- calc_acc(y_val,   prob_val)

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