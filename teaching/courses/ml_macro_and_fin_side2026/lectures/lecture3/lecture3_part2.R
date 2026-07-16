# ==============================================================================
# SIdE Summer Course 2026
# Lecture 3: Tree-Based Models
# Script: Part 2 - Random Forests & Gradient Boosting
# ==============================================================================

# Load required libraries
library(tidyverse)
library(tidymodels)
library(vip)       # For variable importance plots
library(gridExtra) # For arranging plots side-by-side

# ------------------------------------------------------------------------------
# 2.1 Data Prep and Preprocessing Setup
# ------------------------------------------------------------------------------

message("Loading train/validation dataset...")
# Load the recycled recipe (fredmd_recipe) and folds from Lecture 2
load("data/recycled_objects.RData")
treated_data <- read_csv("data/train_val_fredmd.csv")

# ------------------------------------------------------------------------------
# 2.2 Random Forest Specification and Tuning
# ------------------------------------------------------------------------------

# Specify Random Forest model
# We fix mtry at floor(p/3) and tune only min_n to save computational time
rf_spec <- rand_forest(
  mtry = floor(493/3),
  trees = 500,
  min_n = tune()
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("regression")

rf_workflow <- workflow() %>%
  add_recipe(fredmd_recipe) %>%
  add_model(rf_spec)

# Define tuning grid
rf_grid <- grid_regular(
  min_n(range = c(5, 50)),
  levels = 5
)

message("Tuning Random Forest model...")

# library(future)
# plan(multisession, workers = parallel::detectCores() - 4)

set.seed(456)

rf_results <- tune_grid(
  rf_workflow,
  resamples = folds,
  grid = rf_grid,
  metrics = metric_set(rmse, mae),
  control = control_grid(save_pred = TRUE, verbose = TRUE)
)

# plan(sequential)

# Saving the tuned model
message("Saving tuning results to disk...")
save(rf_results, rf_spec, rf_workflow, file = "data/tuned_rf.RData")

# Quick inspection of the hyperparameter tuning process
autoplot(rf_results)

# Select the best RF model
best_rf_params <- select_best(rf_results, metric = "rmse")
message("Best Random Forest parameters:")
print(best_rf_params)

# Finalize and fit the Random Forest model
final_rf_wf <- finalize_workflow(rf_workflow, best_rf_params)
set.seed(456)
final_rf_fit <- fit(final_rf_wf, data = tail(treated_data, 360))

# ------------------------------------------------------------------------------
# 2.3 Gradient Boosting (XGBoost)
# ------------------------------------------------------------------------------

# Specify XGBoost model
xgb_spec <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

xgb_workflow <- workflow() %>%
  add_recipe(fredmd_recipe) %>%
  add_model(xgb_spec)

# Define tuning grid for XGBoost
xgb_grid <- grid_regular(
  trees(range = c(500, 2000)),
  tree_depth(range = c(1, 6)),
  learn_rate(range = c(0.005, 0.1), trans = NULL),
  levels = 3
)

message("Tuning XGBoost model...")
# plan(multisession, workers = parallel::detectCores() - 4)

set.seed(789)

xgb_results <- tune_grid(
  xgb_workflow,
  resamples = folds,
  grid = xgb_grid,
  metrics = metric_set(rmse, mae),
  control = control_grid(save_pred = TRUE, verbose = TRUE)
)

# plan(sequential)

# Saving the tuned model
message("Saving tuning results to disk...")
save(xgb_results, xgb_spec, xgb_workflow, file = "data/tuned_xgb.RData")

# Quick inspection of the hyperparameter tuning process
autoplot(xgb_results)

# Select the best XGBoost model
best_xgb_params <- select_best(xgb_results, metric = "rmse")
message("Best XGBoost parameters:")
print(best_xgb_params)

# Finalize and fit the XGBoost model
final_xgb_wf <- finalize_workflow(xgb_workflow, best_xgb_params)
set.seed(789)
final_xgb_fit <- fit(final_xgb_wf, data = tail(treated_data, 360))

# ------------------------------------------------------------------------------
# 2.4 Variable Importance Metrics (VIM)
# ------------------------------------------------------------------------------

# Extract variable importance from the finalized Random Forest model
message("Plotting variable importance for Random Forest...")
p_rf <- final_rf_fit %>%
  extract_fit_parsnip() %>%
  vip(
    num_features = 20, 
    geom = "col", 
    aesthetics = list(fill = "darkred")
  ) +
  theme_minimal() +
  labs(
    title = "Random Forest",
    subtitle = "Top 20 predictors (Impurity)",
    y = "Importance",
    x = "Predictor"
  )

# Extract variable importance from the finalized XGBoost model
message("Plotting variable importance for XGBoost...")
p_xgb <- final_xgb_fit %>%
  extract_fit_parsnip() %>%
  vip(
    num_features = 20, 
    geom = "col", 
    aesthetics = list(fill = "darkorange")
  ) +
  theme_minimal() +
  labs(
    title = "XGBoost",
    subtitle = "Top 20 predictors (Gain)",
    y = "Importance",
    x = "Predictor"
  )

# Plot both VIMs side-by-side using gridExtra
grid.arrange(p_rf, p_xgb, ncol = 2)
