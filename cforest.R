set.seed(SEED)

results_list <- list()

make_objective <- function(fold_index) {

  function(trials) {

    selected_idx <- trials$suggest_int("subset", 8, min(16, length(FS_cforest_2)))
    feats <- FS_cforest_2[1:selected_idx]

    train_df <- analysis(folds$splits[[fold_index]]) %>%
      dplyr::select(all_of(c("target", feats)))
    val_df   <- assessment(folds$splits[[fold_index]]) %>%
      dplyr::select(all_of(c("target", feats)))
    test_df  <- test_df_15 %>% 
      dplyr::select(all_of(c("target", feats)))

    params <- list(
      ntree = trials$suggest_int("ntree", 200L, 600L),
      mincriterion = 0.95, 
      teststat     = "quadratic",
      testtype     = "Bonferroni",
      replace      = trials$suggest_categorical("replace", c(TRUE, FALSE)),
      fraction     = trials$suggest_uniform("fraction", 0.5, 1),
      cores        = 5,
      minsplit = trials$suggest_int("minsplit", 15, 35),
      minbucket = trials$suggest_int("minbucket", 3, 6),
      maxdepth = trials$suggest_int("maxdepth", 2, 6)                 
    )
    ctrl <- ctree_control(
       teststat = params$teststat,
       testtype = params$testtype,
       mincriterion = params$mincriterion,
       saveinfo = FALSE,
       minsplit = params$minsplit,
       minbucket = params$minbucket,
       maxdepth = params$maxdepth
    )
    mod <- partykit::cforest(
      formula = target ~ .,
      data    = train_df,
      control = ctrl,
      ntree   = params$ntree,
      perturb = list(
        replace  = params$replace,
        fraction = params$fraction
      ),
      cores = params$cores
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