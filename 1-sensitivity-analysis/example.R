source("DIF_simu.R")
source("DIF_sens.R")
source("dif_method_lrt.R") 
source("dif_method_mimic.R")
source("dif_method_raju.R")


## ------------------------------
## Example usage
## ------------------------------

J_A = 100
J_B = 10
J_C = 10
J_D = 10
J = J_A + J_B + J_C + J_D

cfg  <- make_scenario_3(
  N   = 2000,
  J_A = J_A,
  J_B = J_B,
  J_C = J_C,
  J_D = J_D
)
sim  <- simulate_abcd(cfg)

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
## Fit unidimensional 2PL to estimate theta.
## ------------------------------

library(mirt)

dat <- as.data.frame(Y)

mod_2pl <- mirt(
  data     = dat,
  model    = 1,        # 1 latent dimension
  itemtype = "2PL"     # 2-parameter logistic
)

theta_hat <- fscores(
  mod_2pl,
  method = "EAP"
)

## theta_hat_2  would only use type A data, select a subset of Y
## 

theta_hat_vec <- as.numeric(theta_hat[,1])

## ------------------------------
## DIF via difR. What items would be claimed by DIF for this package difR?
## ------------------------------

library(difR)
Data  <- as.data.frame(Y)
group <- Z               # 0 = focal (Spanish), 1 = reference (English)

res_log <- difLogistic(
  Data       = Data,
  group      = group,
  focal.name = 0,        # Spanish is focal group
  type       = "udif",   # uniform + nonuniform DIF
  criterion  = "LRT",
  alpha      = 0.05
)

# Store the p-values
item_pval_difR <- res_log$p.value

# DIF items identified by difR
dif_items <- res_log$DIFitems

## Add more DIF methods, not just difR. Store the detected DIF and the p-value

##IRT-LRT (mirt)
lrt_results <- run_lrt_dif(
  Y        = Y,
  Z        = Z,
  seed     = cfg$seed,
  scenario = "scenario_3"
)
lrt_pvals     <- lrt_results$p_value
lrt_dif_items <- lrt_results$item[lrt_results$flag]



##MIMIC DIF (mirt)
mimic_results <- run_mimic_dif(
  Y        = Y,
  Z        = Z,
  seed     = cfg$seed,
  scenario = "scenario_3"
)
mimic_pvals     <- mimic_results$p_value
mimic_dif_items <- mimic_results$item[mimic_results$flag]

##Raju’s Area DIF
raju_results <- run_raju_dif(
  Y        = Y,
  Z        = Z,
  seed     = cfg$seed,
  scenario = "scenario_3"
)
raju_pvals     <- raju_results$p_value
raju_dif_items <- raju_results$item[raju_results$flag]


sens_results_theta_hat <- lapply(1:J, function(j) {
  binarysens_for_item(
    j              = j,
    Y              = Y,
    Z              = Z,
    theta_vec  = theta_hat_vec,
    Lambda          = 2,       # max sensitivity parameter
    LambdaInc       = 0.01      # increment
  )
})

sens_results_theta <- lapply(1:J, function(j) {
  binarysens_for_item(
    j              = j,
    Y              = Y,
    Z              = Z,
    theta_vec  = theta,
    Lambda          = 2,       # max sensitivity parameter
    LambdaInc       = 0.01      # increment
  )
})

sens_results_theta_hat_df <- do.call(rbind, lapply(sens_results_theta_hat, extract_sens_table))
sens_results_theta_df <- do.call(rbind, lapply(sens_results_theta, extract_sens_table))

## output the critical lambda
crit_theta_hat <- sapply(sens_results_theta_hat, critical_lambda)
names(crit_theta_hat) <- paste0("item", sapply(sens_results_theta_hat, function(r) r$item))
crit_theta <- sapply(sens_results_theta, critical_lambda)
names(crit_theta) <- paste0("item", sapply(sens_results_theta, function(r) r$item))
crit_theta

