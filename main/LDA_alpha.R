#library(MASS)

library(glmnet)
library(gmm)
library(Rcpp)
library(float)
#library(MASS)
sourceCpp("./main/lda_cpp.cpp")


lda_moment = function(X_train,groups_train,X_test,prior=NULL,tol=1e-4)
{

  #--------------consistency check------------

  # ----- data type & integrity -----
  if (is.null(dim(X_train))) stop("X_train is not a matrix")
  X_train = as.matrix(X_train)
  if (any(!is.finite(X_train))) stop("X_train contains NA/NaN/Inf")

  if (missing(X_test)) stop("X_test is missing")
  X_test = as.matrix(X_test)
  if (any(!is.finite(X_test))) stop("X_test contains NA/NaN/Inf")
  if (ncol(X_train) != ncol(X_test)) stop("X_train and X_test have different number of columns")

  # ----- grouping variable -----
  n = nrow(X_train)
  if (n != length(groups_train)) stop("nrow(X_train) != length(groups_train)")
  groups = factor(groups_train)
  lev = levels(groups)
  k = length(lev)
  counts = as.vector(table(groups))
  if (any(counts == 0)) {
    empty = lev[counts == 0]
    warning(sprintf("Empty group(s): %s", paste(empty, collapse = " ")))
    lev = lev[counts > 0L]
    groups = factor(groups, levels = lev)
    k = length(lev)
    counts = as.vector(table(groups))
  }
  if (n <= k) stop("Number of observations <= number of groups")

  # ----- prior probabilities -----
  if (is.null(prior)) {
    prior = counts / n
  } else {
    if (any(prior < 0) || abs(sum(prior) - 1) > tol)
      stop("prior must be non‑negative and sum to 1")
    if (length(prior) != k) stop("length(prior) != number of groups")
  }
  names(prior) = lev


  p = ncol(X_train)

  group_means = rowsum(X_train, groups) / counts
  dimnames(group_means) = list(lev, colnames(X_train))

  centered = X_train - group_means[as.character(groups), , drop = FALSE]

  pooled_var = crossprod(centered)/(n-k)
  pooled_sd = sqrt(diag(pooled_var))

  if (any(pooled_sd < tol)) {
    const = which(pooled_sd < tol)
    stop(sprintf("Variables %s constant within groups",
                 paste(colnames(X_train)[const], collapse = ", ")))
  }

  #--------------Transforming matrix------------------

  scaling_matrix = diag(1/pooled_sd,p,p)
  fac = 1/(n-k)
  X_resid = sqrt(fac) * centered %*% scaling_matrix

  svd_res = svd(X_resid,nu=0L)
  rank = sum(svd_res$d>tol)

  if (rank == 0L) {stop("Rank 0: variables are collinear")}
  if (rank < p) {warning("Variables are collinear; rank = ", rank)}

  scaling_matrix = scaling_matrix %*%
                    svd_res$v[,1:rank] %*%
                    diag(1/svd_res$d[1:rank],rank,rank)


  #-------------start class prediction of x_test-----------------

  X_proj = X_test %*% scaling_matrix
  mu_proj = group_means %*% scaling_matrix
  n_test = nrow(X_test)
  dist2 = matrix(0,nrow = n_test,ncol=k)
  
  for(j in 1:k)
  {
    temp = X_proj - matrix(mu_proj[j,],n_test,rank,byrow=TRUE)
    dist2[,j] = rowSums(temp^2)

  }
  
  scores = 1/2*dist2-matrix(log(prior),n_test,k,byrow=TRUE)

  pred_class = factor(apply(scores,1,which.min),
                      levels=lev)


  #-------------in-sample prediction---------------------------

  X_train_proj = X_train %*% scaling_matrix
  dist2_train = matrix(0,nrow = n,ncol=k)

  for(j in 1:k)
  {
    temp = X_train_proj - matrix(mu_proj[j,],n,rank,byrow=TRUE)
    dist2_train[,j] = rowSums(temp^2)
  }
  scores_train = 1/2*dist2_train-matrix(log(prior),n,k,byrow=TRUE)

  pred_class_train = factor(apply(scores_train,1,which.min),
                            levels=lev)

  train_error_rate = mean(pred_class_train != groups_train)

  return(
    list(
        pred_class = pred_class,
        pred_class_train = pred_class_train,
        train_error = train_error_rate
        )
    )

}



test_error_rate = function(pred, true) mean(pred != true)

lda_bayes_error_rate = function(mu_mat,Sigma,prior=NULL,tol=1e-4,n_sim=5000)
{
  # mu_mat: K x p matrix, each row is true mean vector
  # Sigma: p x p common covariance matrix (positive definite)
  # prior: prior probabilities (default NULL -> equal)
  # tol: tolerance for checking sum(prior)=1
  # n_sim: Monte Carlo sample size for K >= 3
  
  k=nrow(mu_mat)
  p=ncol(mu_mat)
  
  if(is.null(prior))
  {
    prior = rep(1/k,k)
  }else if (abs(sum(prior)-1)>tol)
  {
    stop("Invalid prior distribution, prior not sum to 1")
  }
  
  
  
  if (k == 2)
  {
    
    diff = mu_mat[1, ] - mu_mat[2, ]
    delta = as.numeric(sqrt(diff %*% solve(Sigma, diff)))
    
    if(abs(delta)<tol){stop("Two class too similar, indistinguishable class")}

    err1 = pnorm(-delta/2 - log(prior[1]/prior[2]) / delta)
    err2 = pnorm(-delta/2 + log(prior[1]/prior[2]) / delta)
    
    return(prior[1] * err1 + prior[2] * err2)
    
  } else{
    #-----------Simulate N=5000 data points to estimate the bayes error rate--------------
    
    Sigma_inv = solve(Sigma)
    
    
   
    coef_mat = Sigma_inv %*% t(mu_mat)            
    const = -0.5 * rowSums((mu_mat %*% Sigma_inv) * mu_mat) + log(prior)  
    
    N = n_sim
    y_true = sample(1:k, N, replace = TRUE, prob = prior)
    X = matrix(0, N, p)
    for (i in 1:k) {
      idx = which(y_true == i)
      nk = length(idx)
      
      if (nk > 0) {
        
        X[idx, ] = MASS::mvrnorm(nk, mu = mu_mat[i, ], Sigma = Sigma)
      }
    }
    
    scores = X %*% coef_mat + matrix(const, N, k, byrow = TRUE)
    y_pred = max.col(scores, ties.method = "first")
    
    
    return(mean(y_pred != y_true))

    
  }
  
  
}

fair_select = function(X, groups, min_num=3) {
  lev = levels(factor(groups))
  idx1 = which(groups == lev[1])
  idx2 = which(groups == lev[2])
  
  n1 = length(idx1)
  n2 = length(idx2)
  n = n1 + n2
  p = ncol(X)
  
  X1 = X[idx1, , drop = FALSE]
  X2 = X[idx2, , drop = FALSE]
  
  mu1_hat = colMeans(X1)
  mu2_hat = colMeans(X2)
  
  # Pooled variance for t-statistic
  var1 = apply(X1, 2, var)
  var2 = apply(X2, 2, var)
  Sp2 = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n - 2)
  
  # Two-sample t-statistic
  T_stat = (mu1_hat - mu2_hat) / sqrt(Sp2 * (1/n1 + 1/n2))
  abs_T = abs(T_stat)
  ordered_idx = order(abs_T, decreasing = TRUE)
  
  # Search space for m: 1 to min(p, n-1)
  max_m = min(p, n - 1)
  criteria = numeric(max_m)
  
  # Prepare pooled covariance matrix for eigenvalue calculation
  X1_cent = scale(X1, center = TRUE, scale = FALSE)
  X2_cent = scale(X2, center = TRUE, scale = FALSE)
  Sigma_pool = (crossprod(X1_cent) + crossprod(X2_cent)) / (n - 2)
  
  for (m in 1:max_m) {
    current_features = ordered_idx[1:m]
    
    # Calculate lambda_max for the subset
    if (m == 1) {
      lambda_max = Sigma_pool[current_features, current_features]
    } else {
      Sigma_m = Sigma_pool[current_features, current_features, drop = FALSE]
      lambda_max = max(eigen(Sigma_m, symmetric = TRUE, only.values = TRUE)$values)
    }
    
    T_m = T_stat[current_features]
    sum_T2 = sum(T_m^2)
    
    # FAIR Criterion Formula
    num = n * (sum_T2 + m * (n1 - n2) / n)^2
    den = m * n1 * n2 + n1 * n2 * sum_T2
    criteria[m] = (1 / lambda_max) * (num / den)
  }
  
  optimal_m = which.max(criteria)
  
  if (length(optimal_m) == 0) {
    optimal_m = min(min_num, p) 
  }
  
  return(ordered_idx[1:optimal_m])
}

lda_fair = function(X_train, groups_train, X_test, prior=NULL) {
  
  # 1. Select optimal features using FAIR on training data
  selected_features = fair_select(X_train, groups_train)
  optimal_m = length(selected_features)
  
  # 2. Subset both training and test data (drop = FALSE ensures they stay matrices even if m=1)
  X_train_sub = X_train[, selected_features, drop = FALSE]
  X_test_sub = X_test[, selected_features, drop = FALSE]

  # Fit the MASS::lda model
  if (is.null(prior)) {
    lda_fit = lda(x = X_train_sub, grouping = groups_train)
  } else {
    lda_fit = lda(x = X_train_sub, grouping = groups_train, prior = prior)
  }
  
  # Generate predictions and calculate the training error
  pred_class = predict(lda_fit, newdata = X_test_sub)$class
  pred_class_train = predict(lda_fit, newdata = X_train_sub)$class
  train_error = mean(pred_class_train != groups_train)
  
  # Package it into the 'res' list so the rest of your function works perfectly
  res = list(
    pred_class = pred_class,
    pred_class_train = pred_class_train,
    train_error = train_error
  )
  
  # Append FAIR-specific metadata to the return list
  res$optimal_m = optimal_m
  res$selected_indices = selected_features
  
  return(res)
}

lda_dsda_train = function(X_train, groups_train, prior=NULL) {
  
  # Setup groups and priors
  n = nrow(X_train)
  groups = factor(groups_train)
  lev = levels(groups)
  n1 = sum(groups == lev[1])
  n2 = sum(groups == lev[2])
  
  if (is.null(prior)) prior = c(n1/n, n2/n)
  
  # 1. Recode responses for penalized least squares
  y_coded = ifelse(groups == lev[1], -n/n1, n/n2)
  
  # 2. Fit Lasso and extract coefficients
  cv_fit = cv.glmnet(X_train, y_coded, family = "gaussian", alpha = 1)
  beta_hat = as.numeric(coef(cv_fit, s = "lambda.min"))[-1] # Drop glmnet intercept
  
  # 3. Optimal intercept correction
  mu1 = colMeans(X_train[groups == lev[1], , drop = FALSE])
  mu2 = colMeans(X_train[groups == lev[2], , drop = FALSE])
  
  # Fast 1D projection to compute beta^T * Sigma * beta
  Z_train = as.numeric(X_train %*% beta_hat)
  var1 = var(Z_train[groups == lev[1]])
  var2 = var(Z_train[groups == lev[2]])
  var_pooled = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n - 2)
  
  term1 = -sum((mu1 + mu2) * beta_hat) / 2
  term2_den = sum((mu2 - mu1) * beta_hat)
  
  # Calculate beta0_opt
  if (abs(term2_den) < 1e-6) {
    beta0_opt = 0 
  } else {
    beta0_opt = term1 - (var_pooled / term2_den) * log(prior[1] / prior[2])
  }
  
  # 4. In-sample predictions
  scores_train = Z_train + beta0_opt
  pred_class_train = factor(ifelse(scores_train > 0, lev[2], lev[1]), levels = lev)
  train_error = mean(pred_class_train != groups_train)
  
  return(list(
    beta_hat = beta_hat,
    beta0_opt = beta0_opt,
    lev = lev,
    pred_class_train = pred_class_train,
    train_error = train_error
  ))
}

lda_dsda_test = function(X_test, dsda_model) {
  
  # Extract parameters from trained model
  beta_hat = dsda_model$beta_hat
  beta0_opt = dsda_model$beta0_opt
  lev = dsda_model$lev
  
  # Predictions: Assign to class 2 if x^T * beta + beta0 > 0
  scores_test = as.numeric(X_test %*% beta_hat) + beta0_opt
  pred_class = factor(ifelse(scores_test > 0, lev[2], lev[1]), levels = lev)
  
  return(pred_class)
}

lda_2rs = function(X_train, groups_train, X_test, prior = NULL, rho = 1.0,keep_cols=NULL) {
  X_train = as.matrix(X_train)
  X_test = as.matrix(X_test)
  groups_train = factor(groups_train)
  lev = levels(groups_train)
  
  if (length(lev) != 2) stop("Designed for exactly 2 classes.")
  
  idx1 = which(groups_train == lev[1])
  idx2 = which(groups_train == lev[2])
  n = length(idx1) + length(idx2)
  if (n <= 2) stop("Sample size must be > 2 to retain n-2 components.")
  
  X1 = X_train[idx1, , drop = FALSE]
  X2 = X_train[idx2, , drop = FALSE]
  
  delta_hat = colMeans(X1) - colMeans(X2)
  
  X1_cent = scale(X1, center = TRUE, scale = FALSE)
  X2_cent = scale(X2, center = TRUE, scale = FALSE)
  
  aug_row = sqrt(n * rho) * delta_hat
  A = rbind(X1_cent, X2_cent, aug_row)

  V_full = svd(A, nu = 0)$v
  if(is.null(keep_cols)){keep_cols = min(n - 2, ncol(V_full))}
  U_rho = V_full[, 1:keep_cols, drop = FALSE]
  
  X_train_rot = X_train %*% U_rho
  X_test_rot = X_test %*% U_rho
  
  dsda_model = lda_dsda_train(X_train_rot, groups_train, prior = prior)
  pred_class = lda_dsda_test(X_test_rot, dsda_model)
  
  list(
    pred_class = pred_class,
    pred_class_train = dsda_model$pred_class_train,
    train_error = dsda_model$train_error,
    U_rho = U_rho,
    dsda_model = dsda_model
  )
}

lda_2rs_moment = function(X_train, groups_train, X_test, prior = NULL, rho = 1.0,keep_cols=NULL) {
  X_train = as.matrix(X_train)
  X_test = as.matrix(X_test)
  groups_train = factor(groups_train)
  lev = levels(groups_train)
  
  if (length(lev) != 2) stop("Designed for exactly 2 classes.")
  
  idx1 = which(groups_train == lev[1])
  idx2 = which(groups_train == lev[2])
  n = length(idx1) + length(idx2)
  if (n <= 2) stop("Sample size must be > 2 to retain n-2 components.")
  
  X1 = X_train[idx1, , drop = FALSE]
  X2 = X_train[idx2, , drop = FALSE]
  
  delta_hat = colMeans(X1) - colMeans(X2)
  
  X1_cent = scale(X1, center = TRUE, scale = FALSE)
  X2_cent = scale(X2, center = TRUE, scale = FALSE)
  aug_row = sqrt(n * rho) * delta_hat
  A = rbind(X1_cent, X2_cent, aug_row)
  
  V_full = svd(A, nu = 0)$v
  if(is.null(keep_cols)){keep_cols = min(n - 2, ncol(V_full))}
  U_rho = V_full[, 1:keep_cols, drop = FALSE]
  
  X_train_rot = X_train %*% U_rho
  X_test_rot = X_test %*% U_rho
  
  if (is.null(prior)) {
    lda_fit = MASS::lda(x = X_train_rot, grouping = groups_train)
  } else {
    lda_fit = MASS::lda(x = X_train_rot, grouping = groups_train, prior = prior)
  }
  
  pred_class = predict(lda_fit, newdata = X_test_rot)$class
  pred_class_train = predict(lda_fit, newdata = X_train_rot)$class
  train_error = mean(pred_class_train != groups_train)
  
  list(
    pred_class = pred_class,
    pred_class_train = pred_class_train,
    train_error = train_error,
    U_rho = U_rho,
    lda_model = lda_fit
  )
}

lda_rs_el = function(X_train, groups_train, X_test, rho=1.0,keep_cols=NULL) {
  
  # Format checks
  X_train = as.matrix(X_train)
  X_test = as.matrix(X_test)
  groups_train = factor(groups_train)
  lev = levels(groups_train)
  
  if (length(lev) != 2) stop("Requires exactly 2 classes.")
  
  idx1 = which(groups_train == lev[1])
  idx2 = which(groups_train == lev[2])
  n1 = length(idx1)
  n2 = length(idx2)
  n = n1 + n2
  
  # Step 1: Label encoding
  Y_train = numeric(n)
  Y_train[idx1] = -n / n1
  Y_train[idx2] = n / n2
  
  # Step 2: RS dimensionality reduction
  X1 = X_train[idx1, , drop = FALSE]
  X2 = X_train[idx2, , drop = FALSE]
  
  delta_hat = colMeans(X1) - colMeans(X2)
  
  X1_cent = scale(X1, center = TRUE, scale = FALSE)
  X2_cent = scale(X2, center = TRUE, scale = FALSE)
  aug_row = sqrt(n * rho) * delta_hat
  
  A = rbind(X1_cent, X2_cent, aug_row)
  V_full = svd(A, nu = 0)$v
  
  if(is.null(keep_cols))
  {
    d = max(1, floor(min(n1, n2) / 2))
    d = min(d, ncol(V_full))
    
    U_rho = V_full[, 1:d, drop = FALSE]
  }
  else
  {
    U_rho = V_full[, 1:keep_cols, drop = FALSE]
  }
  
  d = max(1, floor(min(n1, n2) / 2))
  d = min(d, ncol(V_full))
  
  U_rho = V_full[, 1:d, drop = FALSE]
  
  X_train_rot = X_train %*% U_rho
  X_test_rot = X_test %*% U_rho
  
  # Step 3: Empirical Likelihood estimation
  df_train = data.frame(Y = Y_train, X_train_rot)
  
  lm_init = lm(Y ~ ., data = df_train)
  tet0_init = coef(lm_init)
  tet0_init[is.na(tet0_init)] = 0 
  
  fmla = as.formula(paste("Y ~", paste(colnames(df_train)[-1], collapse = " + ")))
  
  # Corrected argument names mapping strictly to docs
  gel_fit = tryCatch({
    gmm::gel(g = fmla, x = X_train_rot, data = df_train, tet0 = tet0_init, type = "EL")
  }, error = function(e) {
    warning("EL algorithm failed to converge. Falling back to OLS estimates.")
    return(list(coefficients = tet0_init))
  })
  
  beta_all = coef(gel_fit)
  beta0_hat = beta_all[1]
  beta_hat = beta_all[-1]
  
  # Step 4: Classification
  scores_train = as.numeric(X_train_rot %*% beta_hat) + beta0_hat
  pred_class_train = factor(ifelse(scores_train > 0, lev[2], lev[1]), levels = lev)
  train_error = mean(pred_class_train != groups_train)
  
  scores_test = as.numeric(X_test_rot %*% beta_hat) + beta0_hat
  pred_class = factor(ifelse(scores_test > 0, lev[2], lev[1]), levels = lev)
  
  return(list(
    pred_class = pred_class,
    pred_class_train = pred_class_train,
    train_error = train_error,
    U_rho = U_rho,
    coefficients = beta_all
  ))
}

lda_moment_cpp = function(X_train, groups_train, X_test, prior = NULL, tol = 1e-4) {
  
  # Ensure 32-bit float storage
  if (!inherits(X_train, "float32")) X_train = float::fl(X_train)
  if (!inherits(X_test, "float32")) X_test = float::fl(X_test)
  
  # Extract internal data structure for C++
  X_train_data = X_train@Data
  X_test_data = X_test@Data
  
  if (ncol(X_train_data) != ncol(X_test_data)) stop("Column mismatch")
  
  # Group logic
  n = nrow(X_train_data)
  groups = factor(groups_train)
  lev = levels(groups)
  k = length(lev)
  counts = as.vector(table(groups))
  
  if (any(counts == 0)) {
    lev = lev[counts > 0L]
    groups = factor(groups, levels = lev)
    k = length(lev)
    counts = as.vector(table(groups))
  }
  
  groups_idx = as.integer(groups) - 1L 
  
  # Prior
  if (is.null(prior)) {
    prior = counts / n
  } else {
    if (any(prior < 0) || abs(sum(prior) - 1) > tol)
      stop("Prior error")
  }
  
  # Dispatch to C++ backend
  res = lda_moment_core_cpp(
    X_train_data = X_train_data,
    groups_train = groups_idx,
    X_test_data = X_test_data,
    prior = prior,
    k = k,
    tol = tol
  )
  
  pred_class = factor(lev[res$pred_class_idx + 1], levels = lev)
  pred_class_train = factor(lev[res$pred_class_train_idx + 1], levels = lev)
  
  return(list(
    pred_class = pred_class,
    pred_class_train = pred_class_train,
    train_error = res$train_error
  ))
}


