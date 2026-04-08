#include <RcppArmadillo.h>
#include <limits>
#include <cmath>

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

// 1. Core LDA Function (Float32 Memory Mapping)
// [[Rcpp::export]]
List lda_moment_core_cpp(IntegerMatrix X_train_data, 
                         IntegerVector groups_train, 
                         IntegerMatrix X_test_data,
                         NumericVector prior,
                         int k, float tol = 1e-4) {
  
  int n = X_train_data.nrow();
  int p = X_train_data.ncol();
  int n_test = X_test_data.nrow();
  
  float* x_train_ptr = (float*) INTEGER(X_train_data);
  float* x_test_ptr = (float*) INTEGER(X_test_data);
  
  arma::fmat X_train(x_train_ptr, n, p, false, true);
  arma::fmat X_test(x_test_ptr, n_test, p, false, true);
  arma::fvec prior_f = Rcpp::as<arma::fvec>(prior);
  
  arma::fvec counts(k, arma::fill::zeros);
  arma::fmat group_means(k, p, arma::fill::zeros);
  
  for (int i = 0; i < n; i++) {
    int g = groups_train[i];
    group_means.row(g) += X_train.row(i);
    counts[g] += 1.0f;
  }
  
  for (int j = 0; j < k; j++) {
    if (counts[j] > 0) group_means.row(j) /= counts[j];
  }
  
  arma::fmat X_resid = X_train; 
  for (int i = 0; i < n; i++) {
    X_resid.row(i) -= group_means.row(groups_train[i]);
  }
  
  arma::fvec pooled_var(p, arma::fill::zeros);
  for(int j = 0; j < p; ++j) {
    pooled_var[j] = arma::dot(X_resid.col(j), X_resid.col(j)) / (n - k);
  }
  
  arma::fvec pooled_sd = arma::sqrt(pooled_var);
  for (int j = 0; j < p; j++) {
    if (pooled_sd[j] < tol) stop("Constant variable detected");
  }
  
  float fac = 1.0f / (n - k);
  float sqrt_fac = std::sqrt(fac);
  for (int j = 0; j < p; j++) {
    X_resid.col(j) *= (sqrt_fac / pooled_sd[j]);
  }
  
  arma::fmat U, V;
  arma::fvec d;
  arma::svd_econ(U, d, V, X_resid); 
  
  int rank = 0;
  for (arma::uword i = 0; i < d.n_elem; i++) {
    if (d[i] > tol) rank++;
  }
  if (rank == 0) stop("Collinear variables");
  
  arma::fmat V_rank = V.cols(0, rank - 1);
  arma::fmat scaling_matrix(p, rank);
  for (int j = 0; j < p; j++) {
    for (int r = 0; r < rank; r++) {
      scaling_matrix(j, r) = V_rank(j, r) / (pooled_sd[j] * d[r]);
    }
  }
  
  arma::fmat X_proj = X_test * scaling_matrix;
  arma::fmat mu_proj = group_means * scaling_matrix;
  arma::fmat dist2(n_test, k);
  
  for (int j = 0; j < k; j++) {
    arma::fmat diff = X_proj;
    diff.each_row() -= mu_proj.row(j);
    dist2.col(j) = arma::sum(arma::square(diff), 1);
  }
  
  IntegerVector pred_class(n_test);
  for(int i = 0; i < n_test; ++i){
    float min_sc = std::numeric_limits<float>::infinity();
    int p_cl = -1;
    for (int j = 0; j < k; j++) {
      float sc = 0.5f * dist2(i, j) - std::log(prior_f[j]);
      if (sc < min_sc) { min_sc = sc; p_cl = j; }
    }
    pred_class[i] = p_cl;
  }
  
  arma::fmat X_train_proj = X_train * scaling_matrix;
  arma::fmat dist2_train(n, k);
  for (int j = 0; j < k; j++) {
    arma::fmat diff = X_train_proj;
    diff.each_row() -= mu_proj.row(j);
    dist2_train.col(j) = arma::sum(arma::square(diff), 1);
  }
  
  int err_count = 0;
  IntegerVector pred_class_train(n);
  for(int i = 0; i < n; ++i){
    float min_sc = std::numeric_limits<float>::infinity();
    int p_cl = -1;
    for (int j = 0; j < k; j++) {
      float sc = 0.5f * dist2_train(i, j) - std::log(prior_f[j]);
      if (sc < min_sc) { min_sc = sc; p_cl = j; }
    }
    pred_class_train[i] = p_cl;
    if(p_cl != groups_train[i]) err_count++;
  }
  
  return List::create(
    Named("pred_class_idx") = pred_class,
    Named("pred_class_train_idx") = pred_class_train,
    Named("train_error") = (float)err_count / n
  );
}

// 2. Bayes Error (Full Covariance Matrix)
// [[Rcpp::export]]
float lda_bayes_error_rate_cpp(arma::fmat mu_mat, arma::fmat Sigma, arma::fvec prior, float tol = 1e-4, int n_sim = 5000) {
  int k = mu_mat.n_rows;
  int p = mu_mat.n_cols;
  
  if (k == 2) {
    arma::fvec diff = mu_mat.row(0).t() - mu_mat.row(1).t();
    arma::fvec invSig_diff = arma::solve(Sigma, diff);
    float delta = std::sqrt(arma::dot(diff, invSig_diff));
    
    if (delta < tol) stop("Classes indistinguishable");
    
    float log_prior_ratio = std::log(prior[0] / prior[1]);
    float d1 = -delta / 2.0f - log_prior_ratio / delta;
    float d2 = -delta / 2.0f + log_prior_ratio / delta;
    
    return prior[0] * (float)R::pnorm(d1, 0.0, 1.0, 1, 0) + prior[1] * (float)R::pnorm(d2, 0.0, 1.0, 1, 0);
  } else {
    int err_count = 0;
    arma::fmat L = arma::chol(Sigma, "lower");
    arma::fmat coef_mat = arma::solve(Sigma, mu_mat.t()).t(); 
    
    arma::fvec const_term(k);
    for(int i = 0; i < k; ++i) {
      const_term[i] = -0.5f * arma::dot(mu_mat.row(i).t(), coef_mat.row(i).t()) + std::log(prior[i]);
    }
    
    for(int s = 0; s < n_sim; ++s) {
      float u = (float)R::runif(0, 1);
      int y_true = 0;
      float cum_prob = prior[0];
      while(u > cum_prob && y_true < k - 1) {
        y_true++;
        cum_prob += prior[y_true];
      }
      
      arma::fvec z(p);
      for(int j = 0; j < p; ++j) z[j] = (float)R::rnorm(0, 1);
      arma::fvec x = mu_mat.row(y_true).t() + L * z;
      
      if ((int)(coef_mat * x + const_term).index_max() != y_true) err_count++;
    }
    return (float)err_count / n_sim;
  }
}

// 3. CS Data Generator (Outputs labeled & shifted Float32 dataset)
// [[Rcpp::export]]
List generate_cs_data_cpp(int n_per_class, arma::fmat mu_mat, float rho) {
  if (rho < 0.0f || rho >= 1.0f) stop("Rho must be in [0,1)");
  
  int k = mu_mat.n_rows;
  int p = mu_mat.n_cols;
  int n = n_per_class * k;
  
  IntegerMatrix out_X(n, p);
  IntegerVector out_y(n);
  arma::fmat X((float*) INTEGER(out_X), n, p, false, true);
  
  float a2 = rho / (1.0f - rho);
  float scale = 1.0f / std::sqrt(1.0f + a2);
  float sd_r = std::sqrt(a2);
  
  int row_idx = 0;
  for (int i = 0; i < k; ++i) {
    for (int r_idx = 0; r_idx < n_per_class; ++r_idx) {
      out_y[row_idx] = i;
      float r = (float)R::rnorm(0, sd_r);
      for (int j = 0; j < p; ++j) {
        X(row_idx, j) = ((float)R::rnorm(0, 1) + r) * scale + mu_mat(i, j);
      }
      row_idx++;
    }
  }
  return List::create(Named("X") = out_X, Named("y") = out_y);
}

// 4. AR1 Data Generator (Outputs labeled & shifted Float32 dataset)
// [[Rcpp::export]]
List generate_ar1_data_cpp(int n_per_class, arma::fmat mu_mat, float rho) {
  int k = mu_mat.n_rows;
  int p = mu_mat.n_cols;
  int n = n_per_class * k;
  
  IntegerMatrix out_X(n, p);
  IntegerVector out_y(n);
  arma::fmat X((float*) INTEGER(out_X), n, p, false, true);
  
  int row_idx = 0;
  for (int i = 0; i < k; ++i) {
    for (int r_idx = 0; r_idx < n_per_class; ++r_idx) {
      out_y[row_idx] = i;
      float prev = (float)R::rnorm(0, 1);
      X(row_idx, 0) = prev + mu_mat(i, 0);
      for (int j = 1; j < p; ++j) {
        float curr = rho * prev + (float)R::rnorm(0, 1);
        X(row_idx, j) = curr + mu_mat(i, j);
        prev = curr;
      }
      row_idx++;
    }
  }
  return List::create(Named("X") = out_X, Named("y") = out_y);
}