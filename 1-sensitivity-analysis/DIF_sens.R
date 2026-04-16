## ------------------------------
## Sensitivity for matched pairs
## ------------------------------

library(optmatch)
library(dplyr)
library(rbounds)

binarysens_for_item <- function(
    j,
    Y,
    Z,
    theta_vec,
    Lambda     = 3,
    LambdaInc  = 0.1
) {
  # Data for this item
  dat_j <- data.frame(
    id = seq_len(nrow(Y)),
    Y  = Z,
    Z  = Y[, j],                 # treatment: 1 = English, 0 = Spanish
    th = theta_vec
  )
  
  # ---- 1. Match on theta_hat using optmatch ----
  dist_obj <- match_on(Z ~ th, data = dat_j)
  pm <- pairmatch(dist_obj, data = dat_j)
  dat_j$pair <- pm
  
  # keep matched units
  dat_m <- dat_j[!is.na(dat_j$pair), ]
  
  # ---- 2. Keep only pairs with exactly 1 treated and 1 control ----
  pair_counts <- dat_m %>%
    group_by(pair) %>%
    summarize(
      n_treated = sum(Z == 1),
      n_control = sum(Z == 0),
      .groups   = "drop"
    )
  
  valid_pairs <- pair_counts %>%
    filter(n_treated == 1, n_control == 1) %>%
    pull(pair)
  
  dat_m2 <- dat_m %>% filter(pair %in% valid_pairs)
  
  # ---- 3. Compute discordant pairs b, c ----
  pair_diffs <- dat_m2 %>%
    arrange(pair, Z) %>%
    group_by(pair) %>%
    summarize(
      Y_treated = Y[Z == 1],
      Y_control = Y[Z == 0],
      diff      = Y_treated - Y_control,
      .groups   = "drop"
    )
  
  # b: treated better (1,0); c: control better (0,1)
  b <- sum(pair_diffs$diff ==  1L)
  c <- sum(pair_diffs$diff == -1L)
  
  x <- min(b, c)
  y <- max(b, c)
  
  sens_obj <- binarysens(
    x         = x,
    y         = y,
    Gamma     = Lambda,
    GammaInc  = LambdaInc
  )
  
  list(
    item      = j,
    b         = b,
    c         = c,
    sens      = sens_obj
  )
}


sens_to_df <- function(s) {
  txt <- capture.output(print(s))
  tab_lines <- txt[grepl("^\\s*[0-9]+\\.", txt)]
  parts <- strsplit(trimws(tab_lines), "\\s+")
  m <- do.call(rbind, parts)
  data.frame(
    Lambda   = as.numeric(m[, 1]),
    p_lower = as.numeric(m[, 2]),
    p_upper = as.numeric(m[, 3])
  )
}

extract_sens_table <- function(res_item) {
  s <- res_item$sens
  df <- sens_to_df(s)
  df$item <- res_item$item
  df$b    <- res_item$b
  df$c    <- res_item$c
  df
}

# if (length(dif_items) == 0) {
#   message("No DIF items detected by difR.")
# } else {
#   # here we probe all B/C/D items: 101–130
#   sens_results <- lapply(1:J, function(j) {
#     binarysens_for_item(
#       j              = j,
#       Y              = Y,
#       Z              = Z,
#       theta_hat_vec  = theta_hat_vec,
#       Lambda          = 2,       # max sensitivity parameter
#       LambdaInc       = 0.1      # increment
#     )
#   })
# }

critical_lambda <- function(res_item) {
  df <- extract_sens_table(res_item)
  idx <- which(df$p_upper > 0.05)
  if (length(idx) == 0) return(Inf)
  df$Lambda[min(idx)]
}



# # some convenience objects if you want to inspect
# if (length(dif_items) > 0) {
#   crit_theta_hat <- sapply(sens_results_theta_hat, critical_lambda)
#   names(crit_theta_hat) <- paste0("item", sapply(sens_results_theta_hat, function(r) r$item))
#   crit_theta_hat[[10]]
#   dif_items
#   sens_df
#   crit_lambdas <- sapply(sens_results, critical_lambda)
# }


