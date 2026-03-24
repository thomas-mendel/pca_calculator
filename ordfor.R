set.seed(SEED)

results_list <- list()

make_objective <- function(fold_index) {

  function(trials) {

    # ---- subset features ----
    selected_idx <- trials$suggest_int("subset", 8, min(16, length(FS_ordfor_2)))
    feats <- FS_ordfor_2[1:selected_idx]

    train_df <- analysis(folds$splits[[fold_index]]) %>% dplyr::select(all_of(c("target", feats)))
    val_df   <- assessment(folds$splits[[fold_index]]) %>% dplyr::select(all_of(c("target", feats)))


    params <- list(
        nsets           = trials$suggest_int("nsets", 50L, 500L),
        ntreeperdiv     = trials$suggest_int("ntreeperdiv", 10L, 60L),
        ntreefinal      = trials$suggest_int("ntreefinal", 1000L, 2000L),
        mtry            = trials$suggest_int("mtry", 2, floor(sqrt(length(feats)))),
        min.node.size   = trials$suggest_int("min.node.size", 10L, 20L),
        perffunction    = "probability",
        nbest           = trials$suggest_int("nbest", 2L, 10L),
        npermtrial      = trials$suggest_int("npermtrial", 10L, 200L),
        classweights    = "balanced",
        always.split.variables = c("PSA"),
        num.threads     = 5
      )


    mod <- ordinalForest::ordfor(
      depvar = "target",
      data   = train_df,
      nsets           = params$nsets,
      ntreeperdiv     = params$ntreeperdiv,
      ntreefinal      = params$ntreefinal,
      mtry            = params$mtry,
      min.node.size   = params$min.node.size,
      perffunction    = params$perffunction,
      nbest           = params$nbest,
      npermtrial      = params$npermtrial,
      classweights    = params$classweights,
      always.split.variables = params$always.split.variables,
      num.threads     = params$num.threads
    )

    prob_train <- predict(mod, newdata = train_df)$classprobs
    prob_val   <- predict(mod, newdata = val_df)$classprobs
    colnames(prob_train) <- levels(train_df$target)
    rownames(prob_train) <- rownames(train_df)
    colnames(prob_val) <- levels(val_df$target)
    rownames(prob_val) <- rownames(val_df)

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