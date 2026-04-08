# Test of extreme dimension using Rcpp

library(Rcpp)
sourceCpp("./main/lda_cpp.cpp")

set.seed(2026)
p = 35000
n_train = 800
n_test = 500
k = 2
prior = rep(1/k, k)
rho = 0

RhpcBLASctl::blas_set_num_threads(4)

mu_mat = matrix(rnorm(k * p, 0, 0.15), k, p)

Sigma = matrix(rho, p, p)
diag(Sigma) = 1

bayes_err = lda_bayes_error_rate_cpp(mu_mat, Sigma, prior)
rm(Sigma)

train_data = generate_cs_data_cpp(n_train / k, mu_mat, rho)
test_data = generate_cs_data_cpp(n_test / k, mu_mat, rho)

res_cpp = lda_moment_core_cpp(
  X_train_data = train_data$X, 
  groups_train = train_data$y, 
  X_test_data = test_data$X, 
  prior = prior, 
  k = k
)

#save(res_cpp,test_data,bayes_err,file="./simulated_datasets/extreme_dimension_case_result/Q4_res_cpp.RData")

