## ------------------------------
## Helpers
## ------------------------------

logistic <- function(x) plogis(x)  # numerically stable logit^{-1}

## items: data.frame with columns a, d, gamma, delta, type
## config: list with fields N, items, eta0, eta_theta, alpha0, alpha_U, alpha_theta, seed

## ------------------------------
## Make A/B/C/D item specs
## ------------------------------

make_abcd_items <- function(
    J_A,
    J_B,
    J_C,
    J_D,
    gamma_B = 0.5,
    delta_C  = 0.5,
    gamma_D = -0.5,  # negative DIF for Type D
    delta_D  = 0.7,   # positive U effect for Type D
    a_range  = c(0.7, 1.5),
    d_range  = c(-1.5, 1.5),
    seed     = NULL
) {
  if (!is.null(seed)) set.seed(seed)
  J <- J_A + J_B + J_C + J_D
  
  a_vals <- runif(J, min = a_range[1], max = a_range[2])
  d_vals <- runif(J, min = d_range[1], max = d_range[2])
  
  type <- c(
    rep("A", J_A),
    rep("B", J_B),
    rep("C", J_C),
    rep("D", J_D)
  )
  
  gamma <- c(
    rep(0,        J_A),
    rep(gamma_B, J_B),   # Type B: pure DIF
    rep(0,        J_C),   # Type C: pure confounding
    rep(gamma_D, J_D)    # Type D: negative DIF
  )
  
  delta <- c(
    rep(0,        J_A),
    rep(0,        J_B),
    rep(delta_C,  J_C),   # Type C: U effect only
    rep(delta_D,  J_D)    # Type D: U effect + DIF
  )
  
  items <- data.frame(
    a      = a_vals,
    d      = d_vals,
    gamma = gamma,
    delta  = delta,
    type   = factor(type, levels = c("A", "B", "C", "D"))
  )
  
  items
}



## ------------------------------
## Core simulator
## ------------------------------

simulate_abcd <- function(config) {
  if (!is.null(config$seed)) set.seed(config$seed)
  
  N     <- config$N
  items <- config$items
  J     <- nrow(items)
  
  ## ---- 1. latent ability theta_i ~ N(0,1)
  theta <- rnorm(N, mean = 0, sd = 1)
  
  ## ---- 2. education U_i | theta_i ~ Bernoulli(logit^{-1}(eta0 + eta_theta * theta_i))
  lin_U <- config$eta0 + config$eta_theta * theta
  p_U   <- logistic(lin_U)
  U     <- rbinom(N, size = 1, prob = p_U)
  
  ## ---- 3. language Z_i | (U_i, theta_i) ~ Bernoulli(logit^{-1}(alpha0 + alpha_U * U_i + alpha_theta * theta_i))
  lin_Z <- config$alpha0 + config$alpha_U * U + config$alpha_theta * theta
  p_Z   <- logistic(lin_Z)
  Z     <- rbinom(N, size = 1, prob = p_Z)
  
  ## ---- 4. item params
  a      <- items$a
  d      <- items$d
  gamma <- items$gamma
  delta  <- items$delta
  
  ## ---- 5. responses Y_ij | theta_i, U_i, Z_i
  ## logit P(Y_ij=1) = a_j * theta_i + d_j + gamma_j * Z_i + delta_j * U_i
  
  lin_Y <- outer(theta, a, "*") +
    matrix(d,      nrow = N, ncol = J, byrow = TRUE) +
    outer(Z,      gamma, "*") +
    outer(U,      delta,  "*")
  
  p_Y <- logistic(lin_Y)
  Y   <- matrix(rbinom(N * J, size = 1, prob = as.vector(p_Y)), nrow = N, ncol = J)
  
  colnames(Y) <- paste0("item", seq_len(J))
  
  list(
    Y          = Y,         # N x J matrix
    theta      = theta,     # length N
    U          = U,         # length N
    Z          = Z,         # length N
    items      = items,     # data.frame
    config     = config
  )
}

## ------------------------------
## Scenario helpers
## ------------------------------

## Scenario 1: no hidden education effect
##   eta_theta = 0, alpha_U = 0, delta_C = 0, delta_D = 0
make_scenario_1 <- function(
    N          = 2000,
    J_A        = 100,
    J_B        = 10,
    J_C        = 10,
    J_D        = 10,
    gamma_B   = 0.5,
    gamma_D   = -0.5,
    seed_items = 123,
    seed_sim   = 1
) {
  items <- make_abcd_items(
    J_A      = J_A,
    J_B      = J_B,
    J_C      = J_C,
    J_D      = J_D,
    gamma_B = gamma_B,
    delta_C  = 0,      # turn off U effect for C in Scenario 1
    gamma_D = gamma_D,
    delta_D  = 0,      # turn off U effect for D in Scenario 1
    seed     = seed_items
  )
  
  list(
    N           = N,
    items       = items,
    eta0        = 0,
    eta_theta   = 0,
    alpha0      = 0,
    alpha_U     = 0,   # U does not affect Z
    alpha_theta = 0,
    seed        = seed_sim
  )
}

## Scenario 2: moderate hidden education effect
##   eta_theta = 0, alpha_U > 0, delta_C > 0, delta_D > 0
make_scenario_2 <- function(
    N          = 2000,
    J_A        = 100,
    J_B        = 10,
    J_C        = 10,
    J_D        = 10,
    gamma_B   = 0.5,
    delta_C    = 0.5,
    gamma_D   = -0.5,
    delta_D    = 0.7,
    alpha_U    = 0.7,
    seed_items = 123,
    seed_sim   = 2
) {
  items <- make_abcd_items(
    J_A      = J_A,
    J_B      = J_B,
    J_C      = J_C,
    J_D      = J_D,
    gamma_B = gamma_B,
    delta_C  = delta_C,
    gamma_D = gamma_D,
    delta_D  = delta_D,
    seed     = seed_items
  )
  
  list(
    N           = N,
    items       = items,
    eta0        = 0,
    eta_theta   = 0,
    alpha0      = 0,
    alpha_U     = alpha_U,
    alpha_theta = 0,
    seed        = seed_sim
  )
}

## Scenario 3: strong hidden education effect
##   eta_theta = 0, alpha_U larger, delta_C, delta_D larger
make_scenario_3 <- function(
    N          = 2000,
    J_A        = 100,
    J_B        = 10,
    J_C        = 10,
    J_D        = 10,
    gamma_B   = 0.5,
    delta_C    = 0.7,
    gamma_D   = -0.5,
    delta_D    = 0.9,
    alpha_U    = 1.2,
    seed_items = 123,
    seed_sim   = 3
) {
  items <- make_abcd_items(
    J_A      = J_A,
    J_B      = J_B,
    J_C      = J_C,
    J_D      = J_D,
    gamma_B = gamma_B,
    delta_C  = delta_C,
    gamma_D = gamma_D,
    delta_D  = delta_D,
    seed     = seed_items
  )
  
  list(
    N           = N,
    items       = items,
    eta0        = 0,
    eta_theta   = 0,
    alpha0      = 0,
    alpha_U     = alpha_U,
    alpha_theta = 0,
    seed        = seed_sim
  )
}


