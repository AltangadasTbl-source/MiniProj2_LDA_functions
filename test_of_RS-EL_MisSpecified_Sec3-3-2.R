# Test of the performance of RS-EL under misspecified model

library(MASS)
# library(glmnet)
# library(gmm)
library(foreach)
library(doParallel)
library(RhpcBLASctl)
library(ggplot2)
library(tidyr)
blas_set_num_threads(1)

source("./main/LDA_alpha.R")

set.seed(2026)

n_train = 100
n_test = 800
p = 1000
SimTimes = 200 
rho_val = 1.0
keep_cols = NULL

num_signal = 20
mu_class1 = c(rep(0.5, num_signal), rep(0, p - num_signal))
mu_class2 = -mu_class1 

Sigma = matrix(0.2, nrow = p, ncol = p)
diag(Sigma) = c(rep(5, 50), rep(1, p - 50))

# Group structure settings
M_groups = 4
q_vec = c(0.95, 0.65, 0.35, 0.15) 
alpha_mat = matrix(rnorm(M_groups * p, mean = 0, sd = 0.5), nrow = M_groups, ncol = p)

# Function to generate grouped cross-sectional data
gen_grouped_data = function(n_total, p_dim, M, q_prob, alpha_m, mu1, mu2, cov_mat) {
  n_per_group = ceiling(n_total / M)
  X_list = list()
  Y_list = c()
  
  for(m in 1:M) {
    y_m = rbinom(n_per_group, 1, q_prob[m])
    y_m[y_m == 0] = 2 
    
    n1 = sum(y_m == 1)
    n2 = sum(y_m == 2)
    
    X1 = if(n1 > 0) mvrnorm(n1, alpha_m[m, ] + mu1, cov_mat) else matrix(0, 0, p_dim)
    X2 = if(n2 > 0) mvrnorm(n2, alpha_m[m, ] + mu2, cov_mat) else matrix(0, 0, p_dim)
    
    X_list[[m]] = rbind(X1, X2)
    Y_list = c(Y_list, rep(1, n1), rep(2, n2))
  }
  
  X_out = do.call(rbind, X_list)
  Y_out = factor(Y_list, levels = c(1, 2))
  
  idx = sample(1:nrow(X_out), n_total)
  return(list(X = X_out[idx, ], Y = Y_out[idx]))
}

# Approximate Best Linear Decision Boundary (Oracle)
cat("Computing large-sample Linear Oracle error...\n")
dat_oracle_train = gen_grouped_data(10000, p, M_groups, q_vec, alpha_mat, mu_class1, mu_class2, Sigma)
dat_oracle_test  = gen_grouped_data(20000, p, M_groups, q_vec, alpha_mat, mu_class1, mu_class2, Sigma)

lda_oracle = lda(dat_oracle_train$X, dat_oracle_train$Y)
oracle_err = mean(predict(lda_oracle, dat_oracle_test$X)$class != dat_oracle_test$Y)
cat("Oracle Linear Error:", round(oracle_err, 4), "\n\n") #0.1231

# Parallel setup
cores = detectCores()
cl = makeCluster(cores)
registerDoParallel(cl)

cat("Starting", SimTimes, "parallel simulations on", cores, "cores...\n")

results = foreach(s = 1:SimTimes, .combine = rbind, 
                  .packages = c("MASS", "glmnet", "gmm", "RhpcBLASctl"),
                  .export = c("lda_dsda_train", "lda_dsda_test", "lda_2rs", 
                              "lda_2rs_moment", "lda_rs_el", "test_error_rate", 
                              "gen_grouped_data", "alpha_mat", "q_vec", "M_groups",
                              "mu_class1", "mu_class2", "Sigma", "rho_val", "keep_cols",
                              "n_train", "n_test", "p")) %dopar% {
                                
                                blas_set_num_threads(1)
                                omp_set_num_threads(1)
                                
                                # Generate grouped data
                                train_dat = gen_grouped_data(n_train, p, M_groups, q_vec, alpha_mat, mu_class1, mu_class2, Sigma)
                                test_dat  = gen_grouped_data(n_test, p, M_groups, q_vec, alpha_mat, mu_class1, mu_class2, Sigma)
                                
                                X_train = train_dat$X; groups_train = train_dat$Y
                                X_test = test_dat$X;   groups_test = test_dat$Y
                                
                                prior_empirical = as.numeric(table(groups_train) / length(groups_train))
                                
                                dsda_test = NA; rs_dsda_test = NA
                                rs_moment_test = NA; rs_el_test = NA; mass_test = NA
                                
                                # 1. DSDA
                                tryCatch({
                                  dsda_model = lda_dsda_train(X_train, groups_train, prior_empirical)
                                  dsda_pred_test = lda_dsda_test(X_test, dsda_model)
                                  dsda_test = test_error_rate(dsda_pred_test, groups_test)
                                }, error = function(e) { NULL })
                                
                                # 2. RS-DSDA
                                tryCatch({
                                  rs_dsda_model = lda_2rs(X_train, groups_train, X_test, prior = prior_empirical, rho = rho_val, keep_cols = keep_cols)
                                  rs_dsda_test = test_error_rate(rs_dsda_model$pred_class, groups_test)
                                }, error = function(e) { NULL })
                                
                                # 3. RS-Moment
                                tryCatch({
                                  rs_moment_model = lda_2rs_moment(X_train, groups_train, X_test, prior = prior_empirical, rho = rho_val, keep_cols = keep_cols)
                                  rs_moment_test = test_error_rate(rs_moment_model$pred_class, groups_test)
                                }, error = function(e) { NULL })
                                
                                # 4. RS-EL
                                tryCatch({
                                  rs_el_model = lda_rs_el(X_train, groups_train, X_test, rho = rho_val)
                                  rs_el_test = test_error_rate(rs_el_model$pred_class, groups_test)
                                }, error = function(e) { NULL })
                                
                                # 5. MASS LDA
                                tryCatch({
                                  lda_fit = lda(X_train, groups_train, prior = prior_empirical)
                                  mass_test = mean(predict(lda_fit, X_test)$class != groups_test)
                                }, error = function(e) { NULL })
                                
                                data.frame(
                                  rep = s,
                                  dsda_test_err = dsda_test,
                                  rs_dsda_test_err = rs_dsda_test,
                                  rs_moment_test_err = rs_moment_test,
                                  rs_el_test_err = rs_el_test,
                                  mass_test_err = mass_test
                                )
                              }

stopCluster(cl)

test_errors_long = pivot_longer(results, 
                                cols = -rep, 
                                names_to = "Model", 
                                values_to = "Error_Rate")

test_errors_long$Model = factor(test_errors_long$Model, 
                                levels = c("dsda_test_err", "rs_dsda_test_err", 
                                           "rs_moment_test_err", "rs_el_test_err", "mass_test_err"),
                                labels = c("DSDA", "RS-DSDA", "RS-Moment", "RS-EL", "MASS LDA"))

test_errors_long = na.omit(test_errors_long)

#save(test_errors_long,file="./simulated_datasets/RS-EL_simulation_results/group_effect_sim_long.RData")

#-----------------------------------------------------------------------------------

load("./simulated_datasets/RS-EL_simulation_results/group_effect_sim_long.RData")

library(ggplot2)
library(tidyr)
library(dplyr)

set.seed(2026)

n_train = 100
n_test = 800
p = 1000
SimTimes = 200 
rho_val = 1.0
keep_cols = NULL
oracle_err = 0.12305

test_errors_long %>% 
  group_by(Model) %>%  
  mutate(error_mean = mean(Error_Rate),
         error_sd = sd(Error_Rate)) %>%
  dplyr::select(Model, error_mean, error_sd) %>%
  unique()

p_boxplot = ggplot(test_errors_long, aes(x = Model, y = Error_Rate, fill = Model)) +
  geom_boxplot(alpha = 0.7, outlier.shape = 16, outlier.size = 1.5) +
  geom_hline(yintercept = oracle_err, linetype = "dashed", color = "red", linewidth = 1) +
  scale_fill_viridis_d(option = "plasma") + 
  labs(title = "Performance under Group-Structured Mixture Data",
       subtitle = paste0("n_train = ", n_train, ", p = ", p, "\nRed dashed line represents Large-Sample Linear Oracle Error: ", round(oracle_err, 4)),
       x = "Classification Method",
       y = "Test Error Rate") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"))

print(p_boxplot)