# Adversarial data generation and testing

library(MASS)
library(glmnet)
source("LDA_alpha.R")

# -------------------------------------------------------------------
# 0. Parameter setting and help functoin
# -------------------------------------------------------------------
set.seed(2026) 
n_train_per_class = 100
n_test_per_class  = 1000
p = 500

# Target Bayes Error : 0.10 (M-distance: 2.5631)
target_error = 0.10
target_delta = -2 * qnorm(target_error) 
n_train = n_train_per_class * 2
n_test = n_test_per_class * 2

cat("========================================================\n")
cat("Generating Dual Adversarial Datasets for LDA Evaluation\n")
cat(sprintf("Target Theoretical Bayes Error strictly controlled at: %.4f\n", target_error))
cat("========================================================\n\n")

# --- Help function：Large sample Monte Carlo Bayes Error ---
verify_bayes_mc = function(mu0, mu1, Sigma, n_mc = 100000, is_ill_conditioned = FALSE, v_min = NULL, lambda_min = NULL) {
  cat(sprintf("  [MC Verification] Generating %d samples for simulation...\n", n_mc))
  n_half = n_mc / 2
  X0 = mvrnorm(n_half, mu0, Sigma)
  X1 = mvrnorm(n_half, mu1, Sigma)
  
  if (!is_ill_conditioned) {
    # For not ill-poissed matrix, using inverse directly: w = Sigma^-1 (mu1 - mu0)
    w = solve(Sigma, mu1 - mu0)
  } else {
    # Ill-poissed: using analytical form to concentrate the signal at v_min
    # c = ||mu1 - mu0||2
    c_val = sqrt(sum((mu1 - mu0)^2))
    w = (c_val / lambda_min) * v_min
  }
  
  # optimal line
  const = as.numeric(0.5 * t(mu0 + mu1) %*% w)
  
 
  score0 = as.numeric(X0 %*% w) - const
  score1 = as.numeric(X1 %*% w) - const
  
  err = (sum(score0 > 0) + sum(score1 < 0)) / n_mc
  return(err)
}

cat("[Generation] Building Scenario 2: Extreme Subspace Hiding Adversary...\n")


Z = matrix(rnorm(p * p), p, p)
V = qr.Q(qr(Z))

# Highly ill poised eigen value setting：First 190 are huge，moderate 191-199 and smallest last eigen value
evals = c(rep(1000, 190), rep(10, p - 191), 0.001)
Lambda = diag(evals)
Sigma_pca = V %*% Lambda %*% t(V)

# Hide the difference completely on the direction of v_min
v_min = V[, p] 
# Mahalanobis distance based scaling factor
c_scale_pca = target_delta * sqrt(evals[p])

mu1_pca = rep(0, p)
mu2_pca = c_scale_pca * v_min

# --- Monte Carlo estimate for Bayes Error ---

delta_actual_pca = c_scale_pca / sqrt(evals[p])
bayes_err_pca = pnorm(-delta_actual_pca / 2)
cat(sprintf("=> Scenario 2 Theoretical Bayes Error (Formula): %.4f\n", bayes_err_pca))

mc_err_pca = verify_bayes_mc(mu1_pca, mu2_pca, Sigma_pca, n_mc = 100000, is_ill_conditioned = TRUE, v_min = v_min, lambda_min = evals[p])
cat(sprintf("=> Scenario 2 Monte Carlo Bayes Error (N=100000): %.4f\n\n", mc_err_pca))

Xtrain_pca = rbind(mvrnorm(n_train_per_class, mu1_pca, Sigma_pca),
                   mvrnorm(n_train_per_class, mu2_pca, Sigma_pca))
Ytrain_pca = rep(c(0, 1), each = n_train_per_class)

Xtest_pca = rbind(mvrnorm(n_test_per_class, mu1_pca, Sigma_pca),
                  mvrnorm(n_test_per_class, mu2_pca, Sigma_pca))
Ytest_pca = rep(c(0, 1), each = n_test_per_class)

idx_pca = sample(1:n_train)
Xtrain = Xtrain_pca[idx_pca, ]
Ytrain = Ytrain_pca[idx_pca]
Xtest  = Xtest_pca
Ytest  = Ytest_pca

#save(Xtrain, Ytrain, file="./simulated_datasets/adversarial_data_generatoin/Train.Rdata")
#save(Xtest,  Ytest,  file="./simulated_datasets/adversarial_data_generatoin/Test.Rdata")

train_pca = list(X = Xtrain, Y = Ytrain)
test_pca  = list(X = Xtest, Y = Ytest)

cat("Datasets successfully generated and saved to ./LDA_adversial_datasets/ \n\n")

# -------------------------------------------------------------------
# 3. Algorithm Adversarial Testing
# -------------------------------------------------------------------

run_evaluation = function(train_data, test_data, scenario_name) {
  cat(sprintf("--- Testing on %s ---\n", scenario_name))
  X_tr = train_data$X; Y_tr = train_data$Y
  X_te = test_data$X;  Y_te = test_data$Y
  
  # 1. Sparse LDA (lda_dsda)
  cat("  -> Running Sparse LDA (lda_dsda)...\n")
  dsda_error = tryCatch({
    mod = lda_dsda_train(X_tr, Y_tr)
    pred = lda_dsda_test(X_te, mod)
    mean(as.numeric(as.character(pred)) != Y_te)
  }, error = function(e) {
    message("     [Failed] Sparse LDA: ", e$message)
    NA
  })
  
  # 2. PCA-based LDA (lda_2rs)
  cat("  -> Running SVD/PCA-based LDA (lda_2rs)...\n")
  rs_error = tryCatch({
    mod = lda_2rs(X_tr, Y_tr, X_te)
    mean(as.numeric(as.character(mod$pred_class)) != Y_te)
  }, error = function(e) {
    message("     [Failed] PCA/SVD LDA: ", e$message)
    NA
  })
  
  # 3. Standard LDA (MASS::lda)
  cat("  -> Running Standard LDA (MASS::lda)...\n")
  mass_error = tryCatch({
    # MASS::lda expects a matrix for x, and a factor for grouping
    mod = MASS::lda(x = X_tr, grouping = factor(Y_tr))
    pred = predict(mod, newdata = X_te)$class
    mean(as.numeric(as.character(pred)) != Y_te)
  }, error = function(e) {
    message("     [Failed] MASS::lda: ", e$message)
    NA
  })
  
  # 4. Empirical Likelihood LDA (lda_rs_el)
  cat("  -> Running Empirical Likelihood LDA (lda_rs_el)...\n")
  el_error = tryCatch({
    mod = lda_rs_el(X_tr, Y_tr, X_te)
    mean(as.numeric(as.character(mod$pred_class)) != Y_te)
  }, error = function(e) {
    message("     [Failed] Empirical Likelihood LDA: ", e$message)
    NA
  })
  
  cat(sprintf("\n  Results for %s:\n", scenario_name))
  cat(sprintf("    Sparse LDA Error: %.4f\n", dsda_error))
  cat(sprintf("    PCA/SVD LDA Error: %.4f\n", rs_error))
  cat(sprintf("    MASS::lda Error: %.4f\n", mass_error))
  cat(sprintf("    Empirical Likelihood Error: %.4f\n\n", el_error))
}


run_evaluation(train_pca, test_pca, "Scenario 2 (Subspace Hiding)")
