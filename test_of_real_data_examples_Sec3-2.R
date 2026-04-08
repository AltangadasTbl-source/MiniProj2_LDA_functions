# Test of Real data Examples

# library(devtools)
# install_github('ramhiser/datamicroarray')

library(datamicroarray)
data('alon', package = 'datamicroarray')
alon$y_label = ifelse(alon$y=="t",1,2) 

X = alon$x
y = alon$y_label
dat_alon = cbind(X,y)

# dat_alon |> dim()
# y|> table()
#----------------------------------------


library(MASS)
library(glmnet)
library(foreach)
library(doParallel)
library(RhpcBLASctl)
library(ggplot2)
library(tidyr)

source("./main/LDA_alpha.R")

blas_set_num_threads(1)
omp_set_num_threads(1)

set.seed(2026)
n_reps = 1000
test_fraction = 0.20
rho_val = 1

cl = makeCluster(4)
registerDoParallel(cl)

results = foreach(i = 1:n_reps, .combine = rbind, .packages = c("MASS", "glmnet"),
                  .export = c("X", "y", "dat_alon", "lda_moment", 
                              "test_error_rate", "lda_dsda_train", "lda_dsda_test",
                              "lda_2rs", "lda_2rs_moment", "rho_val")) %dopar% {
                                
                                # Stratified split
                                idx_1 = which(y == 1)
                                idx_2 = which(y == 2)
                                
                                test_idx_1 = sample(idx_1, size = round(test_fraction * length(idx_1)))
                                test_idx_2 = sample(idx_2, size = round(test_fraction * length(idx_2)))
                                
                                test_idx = c(test_idx_1, test_idx_2)
                                train_idx = setdiff(1:nrow(dat_alon), test_idx)
                                
                                X_train = X[train_idx, ]
                                groups_train = factor(y[train_idx])
                                
                                X_test = X[test_idx, ]
                                groups_test = factor(y[test_idx])
                                
                                # Initialize metrics
                                custom_test = NA; mass_test = NA
                                enet_test = NA; dsda_test = NA
                                rs_dsda_test = NA; rs_moment_test = NA
                                
                                # 1. Custom LDA
                                tryCatch({
                                  res_custom = lda_moment(X_train, groups_train, X_test, prior = NULL)
                                  custom_test = test_error_rate(res_custom$pred_class, groups_test)
                                }, error = function(e) { NULL })
                                
                                # 2. MASS LDA
                                tryCatch({
                                  lda_fit = lda(X_train, groups_train)
                                  pred_mass = predict(lda_fit, X_test)
                                  mass_test = mean(pred_mass$class != groups_test)
                                }, error = function(e) { NULL })
                                
                                # 3. Elastic Net
                                tryCatch({
                                  cv_fit = cv.glmnet(X_train, groups_train, family = "binomial", alpha = 0.5)
                                  pred_enet = predict(cv_fit, newx = X_test, s = "lambda.min", type = "class")
                                  enet_test = mean(as.character(pred_enet) != as.character(groups_test))
                                }, error = function(e) { NULL })
                                
                                # 4. DSDA
                                tryCatch({
                                  dsda_model = lda_dsda_train(X_train, groups_train, prior = NULL)
                                  dsda_pred = lda_dsda_test(X_test, dsda_model)
                                  dsda_test = test_error_rate(dsda_pred, groups_test)
                                }, error = function(e) { NULL })
                                
                                # 5. RS-DSDA
                                tryCatch({
                                  rs_dsda_model = lda_2rs(X_train, groups_train, X_test, prior = NULL, rho = rho_val)
                                  rs_dsda_test = test_error_rate(rs_dsda_model$pred_class, groups_test)
                                }, error = function(e) { NULL })
                                
                                # 6. RS-Moment
                                tryCatch({
                                  rs_moment_model = lda_2rs_moment(X_train, groups_train, X_test, prior = NULL, rho = rho_val)
                                  rs_moment_test = test_error_rate(rs_moment_model$pred_class, groups_test)
                                }, error = function(e) { NULL })
                                
                                data.frame(
                                  rep = i,
                                  custom_test_err = custom_test,
                                  mass_test_err = mass_test,
                                  enet_test_err = enet_test,
                                  dsda_test_err = dsda_test,
                                  rs_dsda_test_err = rs_dsda_test,
                                  rs_moment_test_err = rs_moment_test
                                )
                              }

stopCluster(cl)

#save(results, file="./simulated_datasets/real_data_simulation_result/alon_sim_results.RData")

#----------------------------------------------------------------------------

load("./simulated_datasets/real_data_simulation_result/alon_sim_results.RData")

library(ggplot2)
library(tidyr)


test_errors = results[, c("rep", "custom_test_err", 
                          "mass_test_err", "enet_test_err", "dsda_test_err", 
                          "rs_dsda_test_err", "rs_moment_test_err")]

test_errors_long = pivot_longer(test_errors, 
                                cols = -rep, 
                                names_to = "Model", 
                                values_to = "Error_Rate")

test_errors_long$Model = factor(test_errors_long$Model, 
                                levels = c("mass_test_err", "custom_test_err", 
                                           "enet_test_err", "dsda_test_err", 
                                           "rs_moment_test_err", "rs_dsda_test_err"),
                                labels = c("MASS LDA", "Custom LDA", 
                                           "Elastic Net", "DSDA", 
                                           "RS-Moment", "RS-DSDA"))

test_errors_long = na.omit(test_errors_long)

n_reps = max(test_errors$rep, na.rm=TRUE)
test_fraction = 0.20 

p_box = ggplot(test_errors_long, aes(x = Model, y = Error_Rate, fill = Model)) +
  geom_boxplot(alpha = 0.7, outlier.shape = 16, outlier.size = 1.5) +
  scale_fill_viridis_d(option = "plasma") + 
  labs(title = "Generalization Error on Real Data",
       subtitle = paste0(n_reps, " Monte Carlo Replications (Test Fraction = ", test_fraction * 100, "%)"),
       x = "Classification Method",
       y = "Test Error Rate") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 45, hjust = 1, face = "bold"))

print(p_box)


cat("\n=== Final Benchmark Results (", n_reps, " Replications) ===\n", sep = "")

# Custom LDA
cat("Custom LDA (lda_moment):\n")
cat("  Valid Runs:         ", sum(!is.na(results$custom_test_err)), "/", n_reps, "\n")
cat("  Mean Test Error:    ", round(mean(results$custom_test_err, na.rm = TRUE), 4), "\n")
cat("  SD Test Error:      ", round(sd(results$custom_test_err, na.rm = TRUE), 4), "\n\n")

# MASS LDA
cat("MASS LDA:\n")
cat("  Valid Runs:         ", sum(!is.na(results$mass_test_err)), "/", n_reps, "\n")
cat("  Mean Test Error:    ", round(mean(results$mass_test_err, na.rm = TRUE), 4), "\n")
cat("  SD Test Error:      ", round(sd(results$mass_test_err, na.rm = TRUE), 4), "\n\n")

# Elastic Net
cat("Elastic Net (glmnet):\n")
cat("  Valid Runs:         ", sum(!is.na(results$enet_test_err)), "/", n_reps, "\n")
cat("  Mean Test Error:    ", round(mean(results$enet_test_err, na.rm = TRUE), 4), "\n")
cat("  SD Test Error:      ", round(sd(results$enet_test_err, na.rm = TRUE), 4), "\n\n")

# DSDA
cat("DSDA (Lasso Penalized LDA):\n")
cat("  Valid Runs:         ", sum(!is.na(results$dsda_test_err)), "/", n_reps, "\n")
cat("  Mean Test Error:    ", round(mean(results$dsda_test_err, na.rm = TRUE), 4), "\n")
cat("  SD Test Error:      ", round(sd(results$dsda_test_err, na.rm = TRUE), 4), "\n\n")

# RS-DSDA
cat("RS-DSDA (Rotate-and-Solve DSDA):\n")
cat("  Valid Runs:         ", sum(!is.na(results$rs_dsda_test_err)), "/", n_reps, "\n")
cat("  Mean Test Error:    ", round(mean(results$rs_dsda_test_err, na.rm = TRUE), 4), "\n")
cat("  SD Test Error:      ", round(sd(results$rs_dsda_test_err, na.rm = TRUE), 4), "\n\n")

# RS-Moment
cat("RS-Moment (Rotate-and-Solve MASS LDA):\n")
cat("  Valid Runs:         ", sum(!is.na(results$rs_moment_test_err)), "/", n_reps, "\n")
cat("  Mean Test Error:    ", round(mean(results$rs_moment_test_err, na.rm = TRUE), 4), "\n")
cat("  SD Test Error:      ", round(sd(results$rs_moment_test_err, na.rm = TRUE), 4), "\n\n")
