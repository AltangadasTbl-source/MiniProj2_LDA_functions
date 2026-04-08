# Test of RS-EL LDA on CS structure covariance matrix

library(MASS)
# library(glmnet)
# library(gmm)
library(foreach)
library(doParallel)
library(RhpcBLASctl)
library(ggplot2)
library(tidyr)
library(dplyr)

blas_set_num_threads(1)
source("./main/LDA_alpha.R")

set.seed(2026)

n_train = 100
n_test = 400
p = 800
SimTimes = 200 
k_classes = 2
prior = c(0.5, 0.5)
rho_val = 1.0

# # Define parameter grid
# grid_num_signal = c(10, 20, 40,50,80,90)
# grid_k_large = c(10,20, 50)
# grid_mag_large = c(2,3, 5)

grid_num_signal = c(70,75,80,85,90)
grid_k_large = c(30, 50)
grid_mag_large = c(2)

param_grid = expand.grid(
  num_signal = grid_num_signal,
  k_large = grid_k_large,
  mag_large = grid_mag_large
)

# Setup parallel backend
cores = detectCores()
cl = makeCluster(cores)
registerDoParallel(cl)

cat("Starting grid simulation on", cores, "cores...\n")
cat("Total grid points:", nrow(param_grid), "\n")

master_results = list()

for (i in 1:nrow(param_grid)) {
  ns = param_grid$num_signal[i]
  kg = param_grid$k_large[i]
  mg = param_grid$mag_large[i]
  
  cat(sprintf("\n[Grid %d/%d] num_signal=%d, k_large=%d, mag_large=%.1f\n", 
              i, nrow(param_grid), ns, kg, mg))
  
  # Build signal vectors
  mu_class1 = c(rep(0.2, ns), rep(0, p - ns))
  mu_class2 = -mu_class1 
  mu_mat = rbind(mu_class1, mu_class2)
  
  # Build covariance matrix
  Sigma = matrix(0.5, nrow = p, ncol = p)
  diag(Sigma) = c(rep(mg, kg), rep(0.9, p - kg))
  
  # Calculate theoretical Bayes error
  bayes_err = lda_bayes_error_rate(mu_mat, Sigma, prior)
  
  # Run Monte Carlo iterations
  grid_sim_res = foreach(s = 1:SimTimes, .combine = rbind, 
                         .packages = c("MASS", "glmnet", "gmm", "RhpcBLASctl"),
                         .export = c("lda_dsda_train", "lda_dsda_test", "lda_2rs", 
                                     "lda_2rs_moment", "lda_rs_el", "test_error_rate", 
                                     "mu_mat", "Sigma", "rho_val", "prior", 
                                     "n_train", "n_test", "k_classes")) %dopar% {
                                       
                                       blas_set_num_threads(1)
                                       omp_set_num_threads(1)
                                       
                                       # Generate data
                                       X_train = do.call(rbind, lapply(1:k_classes, function(j) mvrnorm(n_train / k_classes, mu_mat[j, ], Sigma)))
                                       groups_train = factor(rep(1:k_classes, each = n_train / k_classes))
                                       
                                       X_test = do.call(rbind, lapply(1:k_classes, function(j) mvrnorm(n_test / k_classes, mu_mat[j, ], Sigma)))
                                       groups_test = factor(rep(1:k_classes, each = n_test / k_classes))
                                       
                                       # Initialize error holders
                                       dsda_test = NA; rs_dsda_test = NA
                                       rs_moment_test = NA; rs_el_test = NA; mass_test = NA
                                       
                                       # 1. DSDA
                                       tryCatch({
                                         dsda_model = lda_dsda_train(X_train, groups_train, prior)
                                         dsda_pred_test = lda_dsda_test(X_test, dsda_model)
                                         dsda_test = test_error_rate(dsda_pred_test, groups_test)
                                       }, error = function(e) { NULL })
                                       
                                       # 2. RS-DSDA (keep_cols = NULL)
                                       tryCatch({
                                         rs_dsda_model = lda_2rs(X_train, groups_train, X_test, prior = prior, rho = rho_val, keep_cols = NULL)
                                         rs_dsda_test = test_error_rate(rs_dsda_model$pred_class, groups_test)
                                       }, error = function(e) { NULL })
                                       
                                       # 3. RS-Moment (keep_cols = NULL)
                                       tryCatch({
                                         rs_moment_model = lda_2rs_moment(X_train, groups_train, X_test, prior = prior, rho = rho_val, keep_cols = NULL)
                                         rs_moment_test = test_error_rate(rs_moment_model$pred_class, groups_test)
                                       }, error = function(e) { NULL })
                                       
                                       # 4. RS-EL (keep_cols = NULL implied by design or explicitly handled if updated)
                                       tryCatch({
                                         rs_el_model = lda_rs_el(X_train, groups_train, X_test, rho = rho_val)
                                         rs_el_test = test_error_rate(rs_el_model$pred_class, groups_test)
                                       }, error = function(e) { NULL })
                                       
                                       # 5. MASS LDA
                                       tryCatch({
                                         lda_fit = lda(X_train, groups_train, prior = prior)
                                         mass_test = mean(predict(lda_fit, X_test)$class != groups_test)
                                       }, error = function(e) { NULL })
                                       
                                       data.frame(
                                         rep = s,
                                         DSDA = dsda_test,
                                         RS_DSDA = rs_dsda_test,
                                         RS_Moment = rs_moment_test,
                                         RS_EL = rs_el_test,
                                         MASS_LDA = mass_test
                                       )
                                     }
  
  # Append grid parameters and Bayes error
  grid_sim_res$num_signal = ns
  grid_sim_res$k_large = kg
  grid_sim_res$mag_large = mg
  grid_sim_res$bayes_err = bayes_err
  
  master_results[[i]] = grid_sim_res
}

stopCluster(cl)

# Combine all results
df_all = bind_rows(master_results)

# Reshape and aggregate data for plotting
df_long = pivot_longer(df_all, 
                       cols = c("DSDA", "RS_DSDA", "RS_Moment", "RS_EL", "MASS_LDA"),
                       names_to = "Model", 
                       values_to = "Error_Rate")

# Calculate Mean and SD 
df_summary = df_long %>%
  filter(!is.na(Error_Rate)) %>%
  group_by(num_signal, k_large, mag_large, bayes_err, Model) %>%
  summarise(
    mean_err = mean(Error_Rate),
    sd_err = sd(Error_Rate),
    .groups = "drop"
  )

# Format model labels
df_summary$Model = factor(df_summary$Model, 
                          levels = c("DSDA", "RS_DSDA", "RS_Moment", "RS_EL", "MASS_LDA"))

#save(df_summary,file="./simulated_datasets/RS-EL_simulation_results/df_summary_p800_correlated_0.9_0.5,RData")

#-----------------------------------------------------------------------------------------

load("./simulated_datasets/RS-EL_simulation_results/df_summary_p800_correlated_0.9_0.5,RData")
library(ggplot2)
library(tidyr)
library(dplyr)

n_train = 100
n_test = 400
p = 800
SimTimes = 200 
k_classes = 2
prior = c(0.5, 0.5)
rho_val = 1.0

# Extract unique Bayes errors to plot as a connected trend line
df_bayes = df_summary %>%
  dplyr::select(num_signal, k_large, mag_large, bayes_err) %>%
  unique()

# Generate Trend Plot
p_trends = ggplot() +
  # Model trends
  geom_line(data = df_summary, aes(x = num_signal, y = mean_err, color = Model), linewidth = 1) +
  geom_point(data = df_summary, aes(x = num_signal, y = mean_err, color = Model), size = 2.5) +
  geom_ribbon(data = df_summary, aes(x = num_signal, ymin = mean_err - sd_err, ymax = mean_err + sd_err, fill = Model), alpha = 0.15) +
  
  # Bayes error trend (dashed line with empty points)
  geom_line(data = df_bayes, aes(x = num_signal, y = bayes_err), linetype = "dashed", color = "black", linewidth = 0.8) +
  geom_point(data = df_bayes, aes(x = num_signal, y = bayes_err), color = "black", shape = 1, size = 2) +
  
  # Faceting and formatting
  facet_grid(mag_large ~ k_large, 
             labeller = labeller(mag_large = label_both, k_large = label_both)) +
  scale_color_viridis_d(option = "turbo") +
  scale_fill_viridis_d(option = "turbo") +
  labs(
    title = "Generalization Error Trends Across varying sparsity and covariance structures",
    subtitle = paste0("n_train = ", n_train, ", p = ", p, ", SimTimes = ", SimTimes, "\nDashed black line indicates the theoretical Bayes Error"),
    x = "Number of Signal Features (num_signal)",
    y = "Mean Test Error Rate"
  ) +
  theme_bw(base_size = 14) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    strip.background = element_rect(fill = "grey90"),
    strip.text = element_text(face = "bold")
  )

print(p_trends)


