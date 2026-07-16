# ==============================================================================
# SIdE Summer Course 2026
# Lecture 2: Regularized Linear Models & Covariance Shrinkage
# Script: Part 1 - Lasso, Ridge, and Elastic Net for Macro Forecasting
# ==============================================================================

# Load required libraries
library(tidyverse)
library(tidymodels)

# ------------------------------------------------------------------------------
# 1.2 Recycling Data & Setup from Lecture 1
# ------------------------------------------------------------------------------

message("Loading train/validation dataset (train_val_fredmd.csv)...")
# Load the train/validation dataset from Lecture 1 (holding out the last 48 months for final comparison)
treated_data <- read_csv("data/train_val_fredmd.csv")

# Set up rolling window resamples (30 years training, 1 month evaluation)
message("Setting up rolling validation splits...")
folds <- rolling_origin(
  treated_data,
  initial = 360,
  assess = 1,
  cumulative = FALSE
)

# Build preprocessing recipe recycling the pipeline from Lecture 1
fredmd_recipe <- recipe(CPIAUCSL ~ ., data = treated_data) %>%
  step_rm(date) %>%
  step_pca(contains("lag1_"),
           num_comp = 4,
           keep_original_cols = TRUE,
           options = list(center = TRUE, scale. = TRUE),
           prefix = "lag1_PC") %>%
  step_pca(contains("lag2_"),
           num_comp = 4,
           keep_original_cols = TRUE,
           options = list(center = TRUE, scale. = TRUE),
           prefix = "lag2_PC") %>%
  step_pca(contains("lag3_"),
           num_comp = 4,
           keep_original_cols = TRUE,
           options = list(center = TRUE, scale. = TRUE),
           prefix = "lag3_PC") %>%
  step_pca(contains("lag4_"),
           num_comp = 4,
           keep_original_cols = TRUE,
           options = list(center = TRUE, scale. = TRUE),
           prefix = "lag4_PC") %>%
  step_zv(d200811)
# ------------------------------------------------------------------------------
# 1.3 Baseline Model: AR(1) via `{tidymodels}`
# ------------------------------------------------------------------------------

message("Setting up baseline AR(1) model...")
# AR(1) Recipe: We can recycle fredmd_recipe and discard all predictors except the first lag of target (lag1_CPIAUCSL) and the d200811 dummy
ar_recipe <- fredmd_recipe %>%
  step_rm(all_predictors(), -contains("lag1_CPIAUCSL"), -contains("d200811"))
# Inspect the recipe to confirm that we appended the steps
print(ar_recipe)

# For computational efficiency (avoiding unnecessary PCA calculations from the recycled base recipe), we define a new, targeted recipe from scratch:
ar_recipe2 <- recipe(CPIAUCSL ~ lag1_CPIAUCSL + d200811, data = treated_data) %>%
  step_zv(d200811)
print(ar_recipe2)

# Specify OLS linear regression using the {parsnip} interface
ar_spec <- linear_reg() %>%
  set_engine("lm")

# Bundle into a workflow
ar_workflow <- workflow() %>%
  add_recipe(ar_recipe2) %>%
  add_model(ar_spec)
# We can inspect the workflow to see the combined recipe and model specification
print(ar_workflow)

# --- Fitting and Predicting on a Single Resample ---
# 1. Extract the first split and its relevant slots
first_split <- folds$splits[[1]]
train_data <- analysis(first_split)
test_data <- assessment(first_split)

# 2. Fit the workflow (preprocessing + estimation) on the training (analysis) data of this split
ar_fit <- fit(ar_workflow, data = train_data)
# We can inspect the results by simply printing the object
print(ar_fit)
# Or call broom::tidy() to return model coefficients in a tibble format
print(tidy(ar_fit))

# 3. Obtain predictions on the out-of-sample (assessment) data of this split
ar_pred <- predict(ar_fit, new_data = test_data) # Returns a tibble with the predictions
print(ar_pred)

# --- Evaluating across all Resamples ---
# Evaluate out-of-sample performance across all folds
message("Running out-of-sample baseline evaluation across all folds...")
ar_results <- fit_resamples(
  ar_workflow,
  resamples = folds,
  metrics = metric_set(rmse, mae),
  control = control_resamples(save_pred = TRUE, verbose = TRUE)
)

# Out-of-sample metrics
print(collect_metrics(ar_results))

# --- Accessing Predictions and Plotting Comparison ---
# 1. Collect out-of-sample predictions across all folds
ar_preds <- collect_predictions(ar_results)
# Inspecting the object
print(head(ar_preds))

# 2. Join predictions with dates using row indices (.row) from treated_data
ar_preds_with_dates <- ar_preds %>%
  inner_join(
    # We add a row index to treated_data to align with the .row column in predictions
    treated_data %>% mutate(.row = row_number()) %>% select(.row, date),
    by = ".row"
  )

# 3. Generate out-of-sample actual vs. predicted plot
ar_plot <- ar_preds_with_dates %>%
  ggplot(aes(x = date)) +
  geom_line(aes(y = CPIAUCSL, color = "Actual"), linewidth = 0.8) +
  geom_line(aes(y = .pred, color = "Predicted"), linewidth = 0.8, linetype = "dashed") +
  theme_minimal() +
  labs(
    title = "Out-of-Sample Predictions vs. Actual Inflation",
    subtitle = "Rolling Window Forecast Comparison for the AR(1) Baseline",
    x = "Date",
    y = "CPIAUCSL",
    color = "Series"
  ) +
  theme(legend.position = "top")

print(ar_plot)

# ------------------------------------------------------------------------------
# 1.4 Tuning Regularized Regression Models
# ------------------------------------------------------------------------------

message("Configuring regularized regression models...")

# 1. Ridge Regression Spec (mixture = 0)
ridge_spec <- linear_reg(
  penalty = tune(), 
  mixture = 0
) %>%
  set_engine("glmnet")

# 2. Lasso Regression Spec (mixture = 1)
lasso_spec <- linear_reg(
  penalty = tune(), 
  mixture = 1
) %>%
  set_engine("glmnet")

# 3. Elastic Net Regression Spec (both parameters tuned)
elnet_spec <- linear_reg(
  penalty = tune(), 
  mixture = tune()
) %>%
  set_engine("glmnet")

# The translate() function exhibits the (template) code that will actually be sent to the engine
translate(lasso_spec)

# Create grid for Ridge and Lasso (1D)
grid_1d <- grid_regular(
  penalty(),
  levels = 50 # Specifying how many equally spaced values to use in the grid
)

# Create grid for Elastic Net (2D)
grid_2d <- grid_regular(
  penalty(),
  mixture(range = c(0.001, 0.999)), # Avoiding Ridge and Lasso as special cases
  levels = c(10, 5) # We can pass different numbers of evaluations for each hyperparameter
)

# Again, note that a tibble is returned
print(grid_2d)

# 1. Setup Workflows
# We get back to `fredmd_recipe` now
ridge_wf <- workflow() %>% add_recipe(fredmd_recipe) %>% add_model(ridge_spec)
lasso_wf <- workflow() %>% add_recipe(fredmd_recipe) %>% add_model(lasso_spec)
elnet_wf <- workflow() %>% add_recipe(fredmd_recipe) %>% add_model(elnet_spec)

# 2. Tune Ridge Regression
message("Tuning Ridge Regression...")
ridge_results <- tune_grid(
  ridge_wf,
  resamples = folds,
  grid = grid_1d,
  metrics = metric_set(rmse, mae),
  control = control_grid(save_pred = TRUE, verbose = TRUE)
)

# 3. Tune Lasso Regression
message("Tuning Lasso Regression...")
lasso_results <- tune_grid(
  lasso_wf,
  resamples = folds,
  grid = grid_1d,
  metrics = metric_set(rmse, mae),
  control = control_grid(save_pred = TRUE, verbose = TRUE)
)

# 4. Tune Elastic Net Regression
message("Tuning Elastic Net Regression...")
elnet_results <- tune_grid(
  elnet_wf,
  resamples = folds,
  grid = grid_2d,
  metrics = metric_set(rmse, mae),
  control = control_grid(save_pred = TRUE, verbose = TRUE)
)

# --- Optional: Parallel Tuning ---
# To run the grid search in parallel instead of sequentially, uncomment the following block:
#
# library(future)
#
# # Register future parallel backend
# all_cores <- parallel::detectCores()
# # Some cores are left free for system stability
# plan(multisession, workers = all_cores - 4)
#
# # Tune Ridge, Lasso, and Elastic Net in parallel
# ridge_results <- tune_grid(ridge_wf,
#   resamples = folds, grid = grid_1d,
#   metrics = metric_set(rmse, mae),
#   control = control_grid(save_pred = TRUE, verbose = TRUE))
# lasso_results <- tune_grid(lasso_wf,
#   resamples = folds, grid = grid_1d,
#   metrics = metric_set(rmse, mae),
#   control = control_grid(save_pred = TRUE, verbose = TRUE))
# elnet_results <- tune_grid(elnet_wf,
#   resamples = folds, grid = grid_2d,
#   metrics = metric_set(rmse, mae),
#   control = control_grid(save_pred = TRUE, verbose = TRUE))
#
# # Reset to sequential execution when finished
# plan(sequential)

# --- Saving Results ---
message("Saving tuning results to disk...")
# Save baseline and regularized tuning results for future lectures
save(
  ar_results, 
  ridge_results, 
  lasso_results, 
  elnet_results, 
  folds, 
  fredmd_recipe, 
  ar_recipe2,
  ar_spec,
  ar_workflow,
  ridge_spec,
  ridge_wf,
  lasso_spec,
  lasso_wf,
  elnet_spec,
  elnet_wf,
  file = "data/recycled_objects.RData"
)

# ------------------------------------------------------------------------------
# Coefficient Sparsity & Variable Selection
# ------------------------------------------------------------------------------

message("Analyzing Lasso coefficients using a fixed model...")
# 1. Specify a Lasso model with a manually selected penalty (mixture = 1)
lasso_fixed_spec <- linear_reg(
  penalty = 0.000212, 
  mixture = 1
) %>%
  set_engine("glmnet")

fixed_wf <- workflow() %>% 
  add_recipe(fredmd_recipe) %>% 
  add_model(lasso_fixed_spec)

# 2. Fit the model to the train/validation dataset
final_fit <- fit(fixed_wf, data = tail(treated_data, 360))

# 3. Extract the underlying glmnet object and plot the coefficient path
# Calling extract_fit_parsnip() returns the parsnip wrapper, and accessing the $fit
# slot gives us direct access to the raw glmnet object for engine-specific plots.
raw_glmnet <- extract_fit_parsnip(final_fit)$fit
plot(raw_glmnet, xvar = "lambda", label = TRUE)

# 4. Extract and clean coefficient estimates
# We use extract_fit_parsnip() combined with tidy() to return coefficients in a tibble.
coefficients <- extract_fit_parsnip(final_fit) %>%
  tidy() %>%
  filter(term != "(Intercept)") %>%
  mutate(abs_estimate = abs(estimate)) %>%
  arrange(desc(abs_estimate))

# 5. Filter and display non-zero coefficients
non_zero_coefs <- coefficients %>% filter(estimate != 0)
print(head(non_zero_coefs, 20))
# Printing the fraction of non-zero coefficients
print(round(nrow(non_zero_coefs)/nrow(coefficients),3))

# 6. Plot top 20 non-zero coefficients
sparsity_plot <- non_zero_coefs %>%
  slice_head(n = 20) %>%
  ggplot(aes(x = estimate, y = reorder(term, estimate))) +
  geom_col(fill = "steelblue") +
  theme_minimal() +
  labs(
    title = "Top 20 Lasso Coefficient Estimates (Fixed Model)",
    subtitle = "Sparsity and Variable Selection on FRED-MD Predictors",
    x = "Coefficient Estimate",
    y = "Predictor Variable"
  )

print(sparsity_plot)
