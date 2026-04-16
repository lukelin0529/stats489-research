# Set seed for reproducibility
set.seed(12345)

cat("========================================\n")
cat("DIF ANALYSIS WITH SEED = 12345\n")
cat("========================================\n\n")

# Load all necessary libraries upfront
library(mirt)
library(difR)
library(lordif)
library(dplyr)
library(optmatch)
library(rbounds)

# Source utility files
source("App_utils.R")
source("DIF_simu.R")
source("DIF_sens.R")

## ------------------------------
## Simulation Setup
## ------------------------------
J_A = 100
J_B = 10
J_C = 10
J_D = 10
J = J_A + J_B + J_C + J_D

cat("Creating scenario configuration...\n")
cfg <- make_scenario_3(
  N   = 2000,
  J_A = J_A,
  J_B = J_B,
  J_C = J_C,
  J_D = J_D
)

cat("Simulating data...\n")
sim <- simulate_abcd(cfg)

Y     <- sim$Y
theta <- sim$theta
U     <- sim$U
Z     <- sim$Z
items <- sim$items

cat("Y dim:", dim(Y), "\n")
cat("Mean theta:", mean(theta), "\n")
cat("Prop U=1:", mean(U), "\n")
cat("Prop Z=1:", mean(Z), "\n")
print(table(items$type))

## ------------------------------
## Fit 2PL Model
## ------------------------------
cat("\nFitting 2PL model to estimate theta...\n")
dat <- as.data.frame(Y)

mod_2pl <- mirt(
  data     = dat,
  model    = 1,
  itemtype = "2PL"
)

theta_hat <- fscores(mod_2pl, method = "EAP")
theta_hat_vec <- as.numeric(theta_hat[,1])

## ------------------------------
## DIF Method 1: difR Logistic Regression
## ------------------------------
cat("\nRunning difR (Logistic Regression)...\n")
Data  <- as.data.frame(Y)
group <- Z

res_log <- difLogistic(
  Data       = Data,
  group      = group,
  focal.name = 0,
  type       = "udif",
  criterion  = "LRT",
  alpha      = 0.05
)

item_pval_difR <- res_log$p.value
dif_items <- res_log$DIFitems
cat("difR detected", length(dif_items), "DIF items\n")

## ------------------------------
## DIF Method 2: lordif (LENIENT)
## ------------------------------
cat("\nRunning lordif (lenient: minCell=5)...\n")
Data_score <- as.data.frame(Y)
group_score <- Z

res_score_lenient <- lordif(
  Data_score,
  group_score,
  criterion = "Chisqr",
  alpha = 0.05,
  minCell = 5,        # Lenient threshold
  pseudo.R2 = "Nagelkerke"
)

dif_score_items_lenient <- res_score_lenient$DIFitems
cat("lordif detected", length(which(res_score_lenient$flag)), "DIF items\n")

## ------------------------------
## DIF Method 3: Wald Test (mirt)
## ------------------------------
cat("\nRunning Wald test (mirt)...\n")
cat("This will take approximately 40 minutes...\n")

group_factor <- factor(group, levels = c(0, 1), labels = c("Spanish", "English"))

mod_group <- multipleGroup(
  dat,
  model = 1,
  group = group_factor,
  SE = TRUE
)

res_wald <- DIF(
  mod_group,
  which.par = c("a1", "d"),
  scheme = "add_sequential",
  p.adjust = "BH"
)

cat("Wald test detected", nrow(res_wald), "DIF items\n")

## ------------------------------
## DIF Method 4: Mantel-Haenszel
## ------------------------------
cat("\nRunning Mantel-Haenszel...\n")
res_mh <- difMH(
  Data = Data,
  group = group,
  focal.name = 0,
  alpha = 0.05,
  purify = TRUE
)

dif_mh_items <- res_mh$DIFitems
if(is.null(dif_mh_items)) {
  dif_mh_items <- integer(0)
}
cat("MH detected", length(dif_mh_items), "DIF items\n")

## ------------------------------
## DIF Method 5: MIMIC (item-by-item)
## ------------------------------
cat("\nRunning MIMIC DIF (item-by-item)...\n")

mimic_pvals <- rep(NA, ncol(Y))
mimic_items <- integer(0)

for (j in 1:ncol(Y)) {
  
  mimic_model <- paste0(
    "F = 1-", ncol(Y), "\n",
    "F ~ Z\n",
    "item", j, " ~ Z"
  )
  
  fit <- try(
    mirt(
      dat,
      model = mimic_model,
      itemtype = "2PL",
      verbose = FALSE
    ),
    silent = TRUE
  )
  
  if (!inherits(fit, "try-error")) {
    w <- try(DIF(fit, which.par = "d"), silent = TRUE)
    if (!inherits(w, "try-error") && length(w$pvals) > 0) {
      mimic_pvals[j] <- w$pvals[1]
      if (w$pvals[1] < 0.05) mimic_items <- c(mimic_items, j)
    }
  }
}

cat("MIMIC detected", length(mimic_items), "DIF items (unstable)\n")


## ------------------------------
## DIF Method 6: IRT-LRT (VERY SLOW)
## ------------------------------
cat("\nRunning IRT-LRT (this may take 1+ hour)...\n")

lrt_pvals <- rep(NA, ncol(Y))
lrt_items <- integer(0)

mod_base <- multipleGroup(
  dat,
  model = 1,
  group = factor(Z),
  itemtype = "2PL",
  invariance = c("slopes", "intercepts"),
  verbose = FALSE
)

for (j in 1:ncol(Y)) {
  
  mod_free <- multipleGroup(
    dat,
    model = 1,
    group = factor(Z),
    itemtype = "2PL",
    invariance = setdiff(colnames(dat), colnames(dat)[j]),
    verbose = FALSE
  )
  
  lrt <- anova(mod_base, mod_free)
  lrt_pvals[j] <- lrt$p[2]
  if (lrt$p[2] < 0.05) lrt_items <- c(lrt_items, j)
}

cat("IRT-LRT detected", length(lrt_items), "DIF items\n")

##sens for IRT-LRT
## Sensitivity Analysis for IRT-LRT DIF items
## ------------------------------

sens_results_lrt <- lapply(lrt_items, function(j) {
  binarysens_for_item(
    j         = j,
    Y         = Y,
    Z         = Z,
    theta_vec = theta_hat_vec,
    Lambda    = 2,
    LambdaInc = 0.01
  )
})

crit_lambda_lrt <- sapply(sens_results_lrt, critical_lambda)
names(crit_lambda_lrt) <- paste0("item", lrt_items)

crit_lambda_lrt
## ------------------------------
## DIF Method 7: Raju Area (Anchored)
## ------------------------------
cat("\nRunning Raju Area DIF (anchored)...\n")

anchor_items <- which(items$type == "A")
anchor_names <- paste0("item", anchor_items)

mod_raju <- multipleGroup(
  dat,
  model = 1,
  group = factor(Z),
  itemtype = "2PL",
  invariance = c(anchor_names, "free_means", "free_var"),
  verbose = FALSE
)

raju_res <- DIF(
  mod_raju,
  which.par = c("a1", "d"),
  scheme = "add"
)

raju_pvals <- raju_res$pvals
raju_items <- which(raju_pvals < 0.05)

cat("Raju detected", length(raju_items), "DIF items (anchored)\n")

## ------------------------------
## DIF Method 8: SIBTEST
## ------------------------------
cat("\nRunning SIBTEST...\n")

library(difR)

# Single-item SIBTEST
res_sib <- difSIBTEST(
  Data       = Data,
  group      = group,
  focal.name = 0,
  alpha      = 0.05
)

sib_items <- res_sib$DIFitems
sib_pvals <- res_sib$p.value

if (is.null(sib_items)) sib_items <- integer(0)

cat("SIBTEST detected", length(sib_items), "DIF items\n")

## ------------------------------
## DIF Method 9: LASSO DIF
## ------------------------------
cat("\nRunning LASSO DIF...\n")

library(glmnet)

# Total score as matching variable
score <- rowSums(Y)

lasso_coef  <- rep(0, ncol(Y))
lasso_items <- integer(0)

for (j in 1:ncol(Y)) {
  
  yj <- Y[, j]
  
  X <- cbind(
    score = score,
    group = group
  )
  
  fit <- cv.glmnet(
    x      = X,
    y      = yj,
    family = "binomial",
    alpha  = 1
  )
  
  coef_mat <- coef(fit, s = "lambda.min")
  
  group_coef <- coef_mat["group", ]
  
  lasso_coef[j] <- group_coef
  
  if (!is.na(group_coef) && abs(group_coef) > 0) {
    lasso_items <- c(lasso_items, j)
  }
}

# ------------------------------
# Store results
# ------------------------------
lasso_results <- data.frame(
  item   = 1:ncol(Y),
  coef   = lasso_coef,
  flag   = abs(lasso_coef) > 0,
  method = "LASSO"
)

lasso_dif_items <- lasso_results$item[lasso_results$flag]

cat("LASSO detected", length(lasso_dif_items), "DIF items\n")


## ------------------------------
## Summary
## ------------------------------
cat("\n========================================\n")
cat("DIF DETECTION COMPLETE\n")
cat("========================================\n")
cat("difR:", length(dif_items), "items\n")
cat("lordif:", length(which(res_score_lenient$flag)), "items\n")
cat("Wald:", nrow(res_wald), "items\n")
cat("MH:", length(dif_mh_items), "items\n\n")
