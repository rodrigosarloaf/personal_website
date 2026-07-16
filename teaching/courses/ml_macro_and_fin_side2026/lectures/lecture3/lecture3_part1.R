# ==============================================================================
# SIdE Summer Course 2026
# Lecture 3: Tree-Based Models
# Script: Part 1 - Decision Trees & Model Selection in Tidymodels
# ==============================================================================

# Load required libraries
library(tidyverse)
library(tidymodels)
library(rpart.plot)

# ------------------------------------------------------------------------------
# 1.2 Model Specification & Hyperparameter Tuning
# ------------------------------------------------------------------------------

# Specify the decision tree model for regression
message("Specifying decision tree model...")
tree_spec <- decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
  ) %>%
  set_engine("rpart") %>%
  # Setting the mode is required when a package can work with different ML problems
  # Options available are:  "censored regression", "classification", "quantile regression", and "regression"
  set_mode("regression")

# Load the pre-configured recipe and folds from Lecture 2
message("Loading pre-configured recipe, folds, and dataset...")
load("data/recycled_objects.RData")
treated_data <- read_csv("data/train_val_fredmd.csv")

# Verify that fredmd_recipe has been loaded into the workspace
print(fredmd_recipe)

# Bundle preprocessing and model specification into a workflow
tree_workflow <- workflow() %>%
  add_recipe(fredmd_recipe) %>%
  add_model(tree_spec)


# Create a regular grid of hyperparameters to evaluate
message("Designing hyperparameter grid...")
tree_grid <- grid_regular(
  cost_complexity(range = c(0, 0.01), trans = NULL),
  tree_depth(range = c(3, 10)),
  min_n(range = c(5, 20)),
  levels = 3
)
print(tree_grid)

# Optional: Parallel tuning backend for efficiency
# library(future)
# plan(multisession, workers = parallel::detectCores() - 4)

message("Tuning decision tree model over rolling windows...")
tree_results <- tune_grid(
  tree_workflow,
  resamples = folds,
  grid = tree_grid,
  metrics = metric_set(rmse, mae),
  control = control_grid(save_pred = TRUE, verbose = TRUE)
)

# plan(sequential) # Reset if using future parallel execution

# Saving this tuned model for our final comparison next class
message("Saving tuning results to disk...")
save(tree_results, tree_spec, tree_workflow, file = "data/tuned_trees.RData")

# ------------------------------------------------------------------------------
# 1.3 Model Selection & Diagnostics
# ------------------------------------------------------------------------------

# Visualize tuning diagnostics using autoplot method
message("Generating tuning diagnostic plots...")
# 1. Default visualization: plot all metrics across hyperparameters
print(autoplot(tree_results))

# 2. Isolate a specific metric (e.g., RMSE)
print(autoplot(tree_results, metric = "rmse"))

# 3. Focus visual attention on the top-performing grid combinations
# In the current case, we get the same results from #1
print(autoplot(tree_results, select_best = TRUE))

# Display the best hyperparameter configurations
message("Showing best tuning parameters based on RMSE:")
print(show_best(tree_results, metric = "rmse", n = 5))

# Select the absolute best parameter set
best_tree_params <- select_best(tree_results, metric = "rmse")
message("Best parameters selected:")
print(best_tree_params)

# Finalize the workflow by replacing the tune() placeholders with best values
message("Finalizing workflow with best parameters...")
final_tree_wf <- finalize_workflow(tree_workflow, best_tree_params)
print(final_tree_wf)

# Fit the final model to the train/validation dataset
message("Fitting finalized decision tree to full train/validation data...")
final_tree_fit <- fit(final_tree_wf, data = tail(treated_data, 360))
print(final_tree_fit)

# Extract and plot the estimated decision tree
tree_obj <- extract_fit_parsnip(final_tree_fit)$fit
rpart.plot(tree_obj)

# ------------------------------------------------------------------------------
# 1.4 Forecasting
# ------------------------------------------------------------------------------

# Extract validation predictions for the best model configuration
message("Extracting pseudo-out-of-sample and holdout predictions...")
val_predictions <- collect_predictions(tree_results, parameters = best_tree_params)
message("Pseudo-out-of-sample predictions:")
print(head(val_predictions))

# 2. Load the holdout test dataset
holdout_data <- read_csv("data/test_fredmd.csv")

# Predict on the holdout test data using the finalized full fit
holdout_predictions <- predict(final_tree_fit, new_data = holdout_data) %>%
  bind_cols(holdout_data %>% select(date, CPIAUCSL))
message("Holdout sample predictions:")
print(head(holdout_predictions))
