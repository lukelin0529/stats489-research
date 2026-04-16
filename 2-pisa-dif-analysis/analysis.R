# PISA DIF Analysis (US Sample)
# Author: Luke Lin, University of Michigan
#
# Prerequisite: load pisa_us data frame before running this script.
# Data source: PISA 2022 International Database (https://www.oecd.org/pisa/data/)

library(difR)
library(rbounds)
library(dplyr)
library(mirt)
library(optmatch)

# ============================================================
# DATA CLEANING
# ============================================================

# Step 1: Identify reading response columns (end in "R")
response_cols <- grep("R$", colnames(pisa_us), value = TRUE)

# Step 2: Keep only dichotomous items (exactly 2 unique non-NA values)
item_reading <- c()
for (item in response_cols) {
  vals <- unique(na.omit(pisa_us[[item]]))
  if (length(vals) == 2) {
    item_reading <- c(item_reading, item)
  }
}

# Step 3: Create response matrix
response <- pisa_us[, item_reading]

# Step 4: Recode so that 1 = correct, 0 = incorrect (original coding: 1/2)
for (i in 1:ncol(response)) {
  response[, i][response[, i] == 2] <- 0
}

# Step 5: Remove all-NA respondents
ind_na_row <- which(rowSums(is.na(response)) == ncol(response))
if (length(ind_na_row) > 0) response <- response[-ind_na_row, ]

# Step 6: Remove sparse items (< 500 non-missing observations)
min_obs <- 500
keep_cols <- which(colSums(!is.na(response)) > min_obs)
response <- response[, keep_cols]

cat("Final response matrix:", nrow(response), "respondents,", ncol(response), "items\n")

# ============================================================
# GROUPING VARIABLES
# (Define each as 0/1 binary; focal group = 1)
# ============================================================

# Example: ESCS (socioeconomic status) — median split
escs_median <- median(pisa_us$ESCS, na.rm = TRUE)
group_escs <- ifelse(pisa_us$ESCS >= escs_median, 0, 1)  # 1 = Low ESCS (focal)

# Additional grouping variables (uncomment and adapt variable names as needed):
# group_gender  <- ifelse(pisa_us$ST004D01T == 1, 0, 1)   # 1 = Female (focal)
# group_immig   <- ifelse(pisa_us$IMMIG == 1, 0, 1)       # 1 = Immigrant (focal)
# group_hisced  <- ifelse(pisa_us$HISCED >= 3, 0, 1)      # 1 = Lower parental edu
# group_belong  <- ifelse(pisa_us$BELONG >= 0, 0, 1)      # 1 = Low belonging
# group_bullied <- ifelse(pisa_us$BULLIED >= 0, 0, 1)     # 1 = Bullied
# group_repeat  <- ifelse(pisa_us$REPEAT == 0, 0, 1)      # 1 = Grade repeater
# group_single  <- ifelse(pisa_us$FAMSUP == 0, 0, 1)      # 1 = Single parent

# ============================================================
# DIF METHODS (run per grouping variable)
# ============================================================

run_dif_analysis <- function(response, group_var, group_name) {
  cat("\n=== DIF Analysis:", group_name, "===\n")

  # Remove respondents with missing group variable
  keep <- !is.na(group_var)
  resp <- response[keep, ]
  grp  <- group_var[keep]

  # --- Logistic Regression DIF ---
  logistic_results <- difLogistic(resp, group = grp, focal.name = 1,
                                  type = "udif", criterion = "LRT", alpha = 0.05)

  # --- BH Correction ---
  p_adj <- p.adjust(logistic_results$p.value, method = "BH")
  flagged_bh <- which(p_adj < 0.05)

  # --- Mantel-Haenszel ---
  mh_results <- difMH(resp, group = grp, focal.name = 1, purify = TRUE)

  # --- Wald Test (IRT-based via difRaju) ---
  wald_results <- tryCatch(
    difRaju(resp, group = grp, focal.name = 1),
    error = function(e) { cat("  Wald test failed:", conditionMessage(e), "\n"); NULL }
  )

  cat("  Logistic DIF items:", length(logistic_results$DIFitems), "\n")
  cat("  After BH correction:", length(flagged_bh), "\n")
  cat("  MH DIF items:", length(mh_results$DIFitems), "\n")
  if (!is.null(wald_results))
    cat("  Wald DIF items:", length(wald_results$DIFitems), "\n")

  list(
    group        = group_name,
    logistic     = logistic_results,
    p_adj_bh     = p_adj,
    flagged_bh   = flagged_bh,
    mh           = mh_results,
    wald         = wald_results
  )
}

# Run for ESCS grouping
results_escs <- run_dif_analysis(response, group_escs, "ESCS")

# ============================================================
# SENSITIVITY ANALYSIS (Rosenbaum)
# ============================================================

run_sensitivity <- function(response, group_var, flagged_items, group_name,
                            Lambda = 3, LambdaInc = 0.1) {
  cat("\n=== Sensitivity Analysis:", group_name, "===\n")

  keep <- !is.na(group_var)
  resp <- response[keep, ]
  grp  <- group_var[keep]

  # Estimate ability (theta) via 2PL
  mod_2pl <- mirt(as.data.frame(resp), model = 1, itemtype = "2PL", verbose = FALSE)
  theta_hat <- as.numeric(fscores(mod_2pl, method = "EAP")[, 1])

  if (length(flagged_items) == 0) {
    cat("  No flagged items — skipping sensitivity analysis.\n")
    return(NULL)
  }

  sens_results <- lapply(flagged_items, function(j) {
    cat("  Item", j, "\n")
    tryCatch(
      binarysens_for_item(j, Y = resp, Z = grp, theta_vec = theta_hat,
                          Lambda = Lambda, LambdaInc = LambdaInc),
      error = function(e) NULL
    )
  })

  sens_results <- Filter(Negate(is.null), sens_results)
  crit_lambdas <- sapply(sens_results, critical_lambda)
  names(crit_lambdas) <- paste0("item", sapply(sens_results, function(r) r$item))

  cat("  Critical lambda summary:\n")
  print(summary(crit_lambdas))

  list(sens_results = sens_results, crit_lambdas = crit_lambdas)
}

# Sensitivity for BH-flagged ESCS items
sens_escs <- run_sensitivity(
  response     = response,
  group_var    = group_escs,
  flagged_items = results_escs$flagged_bh,
  group_name   = "ESCS"
)

# ============================================================
# HELPER: binarysens wrapper (pair-match on theta, run binarysens)
# ============================================================

binarysens_for_item <- function(j, Y, Z, theta_vec,
                                Lambda = 3, LambdaInc = 0.1) {
  dat_j <- data.frame(id = seq_len(nrow(Y)), Y = Z, Z = Y[, j], th = theta_vec)
  dist_obj <- match_on(Z ~ th, data = dat_j)
  pm <- pairmatch(dist_obj, data = dat_j)
  dat_j$pair <- pm
  dat_m <- dat_j[!is.na(dat_j$pair), ]

  pair_counts <- dat_m %>%
    group_by(pair) %>%
    summarize(n_treated = sum(Z == 1), n_control = sum(Z == 0), .groups = "drop")
  valid_pairs <- pair_counts %>%
    filter(n_treated == 1, n_control == 1) %>%
    pull(pair)
  dat_m2 <- dat_m %>% filter(pair %in% valid_pairs)

  pair_diffs <- dat_m2 %>%
    arrange(pair, Z) %>%
    group_by(pair) %>%
    summarize(Y_treated = Y[Z == 1], Y_control = Y[Z == 0],
              diff = Y_treated - Y_control, .groups = "drop")

  b <- sum(pair_diffs$diff ==  1L)
  c <- sum(pair_diffs$diff == -1L)

  sens_obj <- binarysens(x = min(b, c), y = max(b, c),
                         Gamma = Lambda, GammaInc = LambdaInc)
  list(item = j, b = b, c = c, sens = sens_obj)
}

critical_lambda <- function(res_item) {
  txt <- capture.output(print(res_item$sens))
  tab_lines <- txt[grepl("^\\s*[0-9]+\\.", txt)]
  parts <- strsplit(trimws(tab_lines), "\\s+")
  m <- do.call(rbind, parts)
  df <- data.frame(Lambda = as.numeric(m[, 1]),
                   p_lower = as.numeric(m[, 2]),
                   p_upper = as.numeric(m[, 3]))
  crit <- df$Lambda[df$p_upper > 0.05][1]
  if (is.na(crit)) return(max(df$Lambda))
  crit
}
