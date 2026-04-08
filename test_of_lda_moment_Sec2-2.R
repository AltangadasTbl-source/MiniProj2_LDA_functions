# Test of the custom lda_function based on three examples

# ---------- Case 1 low dimensional case ---------
library(MASS)
source("./main/LDA_alpha.R")
set.seed(2026)

mu_mat = matrix(c(0, 2, 1, 3,-2,-4,-1,3,1.3,3.2,-1.7,-3.8), nrow=2,byrow=TRUE) 
Sigma = diag(6)
prior = c(0.3, 0.7)
bayes_err = lda_bayes_error_rate(mu_mat, Sigma, prior)

n_train = 100
X_train = rbind(mvrnorm(n_train/2, mu_mat[1,], Sigma),
                mvrnorm(n_train/2, mu_mat[2,], Sigma))

groups_train = c(rep(1,n_train/2),rep(2,n_train/2))

n_test = 50
X_test = rbind(mvrnorm(n_test/2, mu_mat[1,], Sigma),
               mvrnorm(n_test/2, mu_mat[2,], Sigma))
groups_test = c(rep(1,n_test/2),rep(2,n_test/2))

res = lda_moment(X_train, groups_train, X_test, prior)
test_err = test_error_rate(res$pred_class, factor(groups_test))

lda_obj = lda(x=X_train,grouping=groups_train,prior=prior)
lda_pred = predict(lda_obj, newdata = X_test)
test_err_builtin = mean(lda_pred$class != groups_test)
train_pred_builtin = predict(lda_obj, newdata = X_train)
train_err_builtin = mean(train_pred_builtin$class != groups_train)


cat("===== Comparison of LDA Methods =====\n")
cat("Bayes error rate (theoretical):", bayes_err, "\n\n")

cat("Custom LDA Function:\n")
cat("  Test error:", test_err, "\n")
cat("  Train error:", res$train_error, "\n\n")

cat("R's Built-in LDA Function (MASS):\n")
cat("  Test error:", test_err_builtin, "\n")
cat("  Train error:", train_err_builtin, "\n\n")


# ---------- Case 2 Iris data ----------

# Load required libraries
# library(MASS)  
# source("./main/LDA_alpha.R")  

set.seed(2026)
n_reps <- 100
n_train <- 100  

train_err_custom <- numeric(n_reps)
test_err_custom <- numeric(n_reps)
train_err_mass <- numeric(n_reps)
test_err_mass <- numeric(n_reps)

data(iris)
X <- as.matrix(iris[, 1:4])
y <- as.numeric(iris$Species)  # 1,2,3
n_total <- nrow(X)

for (i in 1:n_reps) {

  train_idx <- sample(1:n_total, n_train)
  X_train <- X[train_idx, ]
  y_train <- y[train_idx]
  X_test <- X[-train_idx, ]
  y_test <- y[-train_idx]
  
  # Custom LDA
  custom_res <- lda_moment(X_train, y_train, X_test)
  train_err_custom[i] <- custom_res$train_error
  test_err_custom[i] <- mean(custom_res$pred_class != y_test)
  
  # MASS LDA
  mass_model <- lda(x = X_train, grouping = y_train)
  train_pred_mass <- predict(mass_model, X_train)$class
  test_pred_mass <- predict(mass_model, X_test)$class
  train_err_mass[i] <- mean(train_pred_mass != y_train)
  test_err_mass[i] <- mean(test_pred_mass != y_test)
}

cat("===== Iris: 100 random 100/50 splits =====\n")
cat("Custom LDA:\n")
cat(sprintf("  Train error: %.4f (sd %.4f)\n", mean(train_err_custom), sd(train_err_custom)))
cat(sprintf("  Test error:  %.4f (sd %.4f)\n", mean(test_err_custom), sd(test_err_custom)))
cat("MASS LDA:\n")
cat(sprintf("  Train error: %.4f (sd %.4f)\n", mean(train_err_mass), sd(train_err_mass)))
cat(sprintf("  Test error:  %.4f (sd %.4f)\n", mean(test_err_mass), sd(test_err_mass)))



# ---------- Case 3 High dimensional case ----------

set.seed(2026)
p=100

mu_mat = matrix(c(rep(0,p),rep(0.15,p)), nrow=2,byrow=TRUE) 
Sigma = diag(p)

prior = c(0.5, 0.5)
bayes_err = lda_bayes_error_rate(mu_mat, Sigma, prior)

n_train = 80
X_train = rbind(mvrnorm(n_train/2, mu_mat[1,], Sigma),
                mvrnorm(n_train/2, mu_mat[2,], Sigma))

groups_train = c(rep(1,n_train/2),rep(2,n_train/2))

n_test = 50
X_test = rbind(mvrnorm(n_test/2, mu_mat[1,], Sigma),
               mvrnorm(n_test/2, mu_mat[2,], Sigma))
groups_test = c(rep(1,n_test/2),rep(2,n_test/2))

res = lda_moment(X_train, groups_train, X_test, prior)
test_err = test_error_rate(res$pred_class, factor(groups_test))

lda_obj = lda(x=X_train,grouping=groups_train,prior=prior)
lda_pred = predict(lda_obj, newdata = X_test)
test_err_builtin = mean(lda_pred$class != groups_test)
train_pred_builtin = predict(lda_obj, newdata = X_train)
train_err_builtin = mean(train_pred_builtin$class != groups_train)


cat("===== Comparison of LDA Methods =====\n")

cat("Bayes error rate (theoretical):", bayes_err, "\n\n")

cat("Custom LDA Function:\n")
cat("  Test error:", test_err, "\n")
cat("  Train error:", res$train_error, "\n\n")

cat("R's Built-in LDA Function (MASS):\n")
cat("  Test error:", test_err_builtin, "\n")
cat("  Train error:", train_err_builtin, "\n\n")









