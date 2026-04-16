# ---------------------------
# Run ID & reproducibility
# ---------------------------
run_id <- as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID", unset = 1))
set.seed(1000 + run_id)

cat("Running simulation with run_id =", run_id, "\n")

library(dplyr)
library(optmatch)
library(rbounds)
library(mirt)

source("DIF_simu.R")


cat("\n=== Step 1: Define Critical Lambda Function ===\n")

# Function to find the "tipping point" lambda
critical_lambda <- function(res_item) {
  # Extract sensitivity table
  sens_df <- sens_to_df(res_item$sens)
  
  # Find the smallest Lambda where p_upper > 0.05
  # (i.e., where we can no longer reject the null hypothesis)
  crit <- sens_df$Lambda[sens_df$p_upper > 0.05][1]
  
  # If never exceeds 0.05, return max Lambda tested
  if (is.na(crit)) {
    return(max(sens_df$Lambda))
  }
  
  return(crit)
}

cat("Critical lambda function defined.\n")

cat("\n=== Running simulation ===\n")

# Item counts
J_A <- 100
J_B <- 10
J_C <- 10
J_D <- 10

# Build scenario
cfg <- make_scenario_3(
  N   = 2000,
  J_A = J_A,
  J_B = J_B,
  J_C = J_C,
  J_D = J_D
)

# Simulate data
sim <- simulate_abcd(cfg)

Y     <- sim$Y
Z     <- sim$Z
theta <- sim$theta
J     <- ncol(Y)

cat("\n=== Estimating theta_hat using all items ===\n")
dat <- as.data.frame(Y)

mod_2pl <- mirt(
  data     = dat,
  model    = 1,
  itemtype = "2PL"
)

theta_hat <- fscores(mod_2pl, method = "EAP")
theta_hat_vec <- as.numeric(theta_hat[, 1])


## ============================================================================
## ANALYSIS 1: Sensitivity Analysis with ESTIMATED theta (all 130 items)
## ============================================================================

cat("\n=== Step 2A: Run Sensitivity Analysis with theta_hat (all items) ===\n")

sens_results_theta_hat <- lapply(1:J, function(j) {
  if(j %% 10 == 0) cat("  Processing item", j, "of", J, "\n")
  binarysens_for_item(
    j              = j,
    Y              = Y,
    Z              = Z,
    theta_vec      = theta_hat_vec,
    Lambda         = 2,
    LambdaInc      = 0.01
  )
})

# Extract results
sens_results_theta_hat_df <- do.call(rbind, lapply(sens_results_theta_hat, extract_sens_table))
crit_theta_hat <- sapply(sens_results_theta_hat, critical_lambda)
names(crit_theta_hat) <- paste0("item", sapply(sens_results_theta_hat, function(r) r$item))

cat("\n=== CRITICAL LAMBDA SUMMARY (Estimated Theta - All Items) ===\n")
print(summary(crit_theta_hat))


## ============================================================================
##ANALYSIS 2: Sensitivity Analysis with TRUE (Observed) theta
## ============================================================================
cat("\n=== Step 2B: Run Sensitivity Analysis with TRUE theta (all items) ===\n")

sens_results_theta_true <- lapply(1:J, function(j) {
  if (j %% 10 == 0) cat("  Processing item", j, "of", J, "\n")
  binarysens_for_item(
    j              = j,
    Y              = Y,
    Z              = Z,
    theta_vec      = theta,     # <-- TRUE theta from simulation
    Lambda         = 2,
    LambdaInc      = 0.01
  )
})

# Extract results
sens_results_theta_true_df <- do.call(
  rbind,
  lapply(sens_results_theta_true, extract_sens_table)
)

crit_theta_true <- sapply(sens_results_theta_true, critical_lambda)
names(crit_theta_true) <- paste0(
  "item",
  sapply(sens_results_theta_true, function(r) r$item)
)

cat("\n=== CRITICAL LAMBDA SUMMARY (TRUE Theta - All Items) ===\n")
print(summary(crit_theta_true))

## ============================================================================
##ANALYSIS 3: Sensitivity Analysis with ANCHOR-ONLY Estimated theta (Type A only)
## ============================================================================
cat("\n=== Estimating theta using anchor-only items (Type A) ===\n")

anchor_items <- 1:J_A   # J_A = 100 by design
dat_anchor <- as.data.frame(Y[, anchor_items])

mod_anchor <- mirt(
  data     = dat_anchor,
  model    = 1,
  itemtype = "2PL"
)

theta_hat_anchor <- fscores(mod_anchor, method = "EAP")
theta_hat_anchor_vec <- as.numeric(theta_hat_anchor[, 1])

cat("\n=== Step 2C: Run Sensitivity Analysis with ANCHOR-ONLY theta (all items) ===\n")

sens_results_theta_anchor <- lapply(1:J, function(j) {
  if (j %% 10 == 0) cat("  Processing item", j, "of", J, "\n")
  binarysens_for_item(
    j              = j,
    Y              = Y,
    Z              = Z,
    theta_vec      = theta_hat_anchor_vec,
    Lambda         = 2,
    LambdaInc      = 0.01
  )
})


# Extract results
sens_results_theta_anchor_df <- do.call(
  rbind,
  lapply(sens_results_theta_anchor, extract_sens_table)
)

crit_theta_anchor <- sapply(sens_results_theta_anchor, critical_lambda)
names(crit_theta_anchor) <- paste0(
  "item",
  sapply(sens_results_theta_anchor, function(r) r$item)
)

cat("\n=== CRITICAL LAMBDA SUMMARY (ANCHOR-ONLY Theta - All Items) ===\n")
print(summary(crit_theta_anchor))

results <- list(
  run_id = run_id,
  crit_theta_hat    = crit_theta_hat,
  crit_theta_true   = crit_theta_true,
  crit_theta_anchor = crit_theta_anchor
)

dir.create("results", showWarnings = FALSE)

saveRDS(
  results,
  file = paste0("results/sens_run_", run_id, ".rds")
)

cat("Saved results for run", run_id, "\n")

