# ==============================================================================
# Lecture 4 - Deep Learning & Model Comparison
# Part 1: Deep Learning in the Tidymodels Environment
# ==============================================================================

library(tidyverse)
library(tidymodels)
library(brulee)

# ------------------------------------------------------------------------------
# 1. Model Specification
# ------------------------------------------------------------------------------

# 1. Single-layer MLP specification
mlp_1_spec <- mlp(
  epochs = 50, # Reduced number for training efficiency
  hidden_units = tune(),
  penalty = tune(),
  learn_rate = tune(),
  activation = tune()
) %>%
  # Turning off the validation - randomly assigns 0.1 by default
  set_engine("brulee", validation = 0) %>%
  set_mode("regression")

print(mlp_1_spec)

# 2. Two-layer MLP specification
mlp_2_spec <- mlp(
  epochs = 50,
  hidden_units = tune(),
  penalty = tune(),
  learn_rate = tune(),
  activation = tune()
) %>%
  # Note how `hidden_units_2` is passed below
  set_engine("brulee_two_layer", hidden_units_2 = tune(), validation = 0) %>%
  set_mode("regression")

print(mlp_2_spec)

# ------------------------------------------------------------------------------
# 2. Workflow Bundle
# ------------------------------------------------------------------------------

# Load pre-saved objects from earlier lectures
load("data/recycled_objects.RData")
treated_data <- read_csv("data/train_val_fredmd.csv")

# Append input standardization to the existing recipe
mlp_recipe <- fredmd_recipe %>%
  step_normalize(all_predictors())

# Workflow for single-layer MLP
mlp_1_workflow <- workflow() %>%
  add_recipe(mlp_recipe) %>%
  add_model(mlp_1_spec)

# Workflow for two-layer MLP
mlp_2_workflow <- workflow() %>%
  add_recipe(mlp_recipe) %>%
  add_model(mlp_2_spec)

print(mlp_1_workflow)
print(mlp_2_workflow)

# ------------------------------------------------------------------------------
# 3. Grid Design
# ------------------------------------------------------------------------------

# Random grid design for single-layer MLP
set.seed(333)
mlp_1_grid <- grid_random(
  hidden_units(),
  penalty(),
  learn_rate(),
  activation(values = c("relu", "tanh", "sigmoid")),
  size = 10
)

# Random grid design for two-layer MLP
set.seed(444)
mlp_2_grid <- grid_random(
  hidden_units(),
  hidden_units_2 = hidden_units(),
  penalty(),
  learn_rate(),
  activation(values = c("relu", "tanh", "sigmoid")),
  size = 10
)

print("Single-layer Tuning Grid:")
print(mlp_1_grid)
print("Two-layer Tuning Grid:")
print(mlp_2_grid)

# ------------------------------------------------------------------------------
# 4. Grid Execution
# ------------------------------------------------------------------------------

# Optional parallel execution
# library(future)
# plan(multisession, workers = parallel::detectCores() - 4)

message("Starting grid tuning for single-layer MLP...")
set.seed(333)
mlp_1_results <- tune_grid(
  mlp_1_workflow,
  resamples = folds,
  grid = mlp_1_grid,
  metrics = metric_set(rmse, mae),
  control = control_grid(save_pred = TRUE, verbose = TRUE)
)

message("Starting grid tuning for two-layer MLP...")
set.seed(444)
mlp_2_results <- tune_grid(
  mlp_2_workflow,
  resamples = folds,
  grid = mlp_2_grid,
  metrics = metric_set(rmse, mae),
  control = control_grid(save_pred = TRUE, verbose = TRUE)
)

# plan(sequential)

# Save the tuned model objects
save(mlp_1_results, mlp_2_results, mlp_1_spec, mlp_2_spec, mlp_1_workflow, mlp_2_workflow, mlp_recipe, file = "data/tuned_mlps.RData")
message("Tuning complete. Results saved to 'data/tuned_mlps.RData'.")

# ------------------------------------------------------------------------------
# 5. Model Selection & Diagnostics
# ------------------------------------------------------------------------------

# Plot metrics across hyperparameters for the two-layer network
autoplot(mlp_2_results)

# Select the best configuration for both architectures
best_mlp_1 <- select_best(mlp_1_results, metric = "rmse")
print("Best Single-layer Parameters:")
print(best_mlp_1)

best_mlp_2 <- select_best(mlp_2_results, metric = "rmse")
print("Best Two-layer Parameters:")
print(best_mlp_2)

# Extract the best validation RMSE for each model structure
rmse_1 <- show_best(mlp_1_results, metric = "rmse", n = 1)$mean
rmse_2 <- show_best(mlp_2_results, metric = "rmse", n = 1)$mean

# Select and finalize the workflow with the lowest validation RMSE
if (rmse_1 < rmse_2) {
  message("Single-layer MLP performed best. Finalizing mlp_1_workflow...")
  final_mlp_wf <- finalize_workflow(mlp_1_workflow, best_mlp_1)
} else {
  message("Two-layer MLP performed best. Finalizing mlp_2_workflow...")
  final_mlp_wf <- finalize_workflow(mlp_2_workflow, best_mlp_2)
}

# Fit the chosen finalized workflow to the full train/validation dataset
set.seed(999)
final_mlp_fit <- fit(final_mlp_wf, data = tail(treated_data, 360))
print("Model Fit Summary:")
print(final_mlp_fit)
