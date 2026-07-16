# ==============================================================================
# SIdE Summer Course 2026
# Lecture 4: Deep Learning & Model Comparison
# Script: Part 2 - Model Comparison & Wrapping Up
# ==============================================================================

# Load required libraries
library(tidyverse)
library(tidymodels)
library(forecast)
library(yardstick)

# ------------------------------------------------------------------------------
# 2.2 Out-of-Sample Performance Comparison
# ------------------------------------------------------------------------------

# ----- Tab 0: Auxiliary Forecasting Function -----
message("Defining auxiliary forecasting function...")

# Define an auxiliary helper function to finalize workflows and generate rolling out-of-sample predictions
finalize_and_predict <- function(spec, recipe, results = NULL, last_train_data, test_data, seed) {
  wf <- workflow() %>%
    add_recipe(recipe) %>%
    add_model(spec)
  
  if (!is.null(results)) {
    best_params <- select_best(results, metric = "rmse")
    wf <- finalize_workflow(wf, best_params)
  }
  
  n_test <- nrow(test_data)
  window_size <- nrow(last_train_data)
  predictions <- numeric(n_test)
  
  # Combine datasets to maintain a fixed-size rolling window
  combined_data <- bind_rows(last_train_data, test_data)
  
  # Note that we could have used rolling_origin() and fit_resamples() here as well
  set.seed(seed)
  for (i in 1:n_test) {
    # Fixed-size rolling training window
    active_train <- combined_data[i:(window_size + i - 1), ]
    
    fit_model <- fit(wf, data = active_train)
    predictions[i] <- predict(fit_model, new_data = test_data[i, ]) %>% pull(.pred)
  }
  
  return(predictions)
}

# ----- Tab 1: Predictions -----
message("Loading out-of-sample datasets and saved tuning results...")

# Load holdout test dataset and training data
holdout_data <- read_csv("data/test_fredmd.csv")
treated_data <- read_csv("data/train_val_fredmd.csv")
treated_data <- tail(treated_data, 360) # Last train data used during train/validation

message("Generating rolling out-of-sample forecasts for all models...")

# 1. Baseline AR(1) Forecast (re-estimated at each step)
load("data/recycled_objects.RData") # Load saved tuning results and recipes
pred_ar1 <- finalize_and_predict(ar_spec, ar_recipe2, results = NULL, treated_data, holdout_data, seed = 1)
rm(ar_spec, ar_recipe2, ar_workflow, ar_results, folds) # Removing previous objects after using to free memory

# 2. Random Walk (RW) Forecast
pred_rw <- holdout_data$lag1_CPIAUCSL

# 3. Lasso Regularized Linear Forecast
pred_lasso <- finalize_and_predict(lasso_spec, fredmd_recipe, lasso_results, treated_data, holdout_data, seed = 2)
rm(lasso_spec, lasso_wf, lasso_results)

# 4. Ridge Regularized Linear Forecast
pred_ridge <- finalize_and_predict(ridge_spec, fredmd_recipe, ridge_results, treated_data, holdout_data, seed = 3)
rm(ridge_spec, ridge_wf, ridge_results)

# 5. Elastic Net Regularized Linear Forecast
pred_elnet <- finalize_and_predict(elnet_spec, fredmd_recipe, elnet_results, treated_data, holdout_data, seed = 4)
rm(elnet_spec, elnet_wf, elnet_results)

# 6. Decision Tree Forecast
load("data/tuned_trees.RData")
pred_tree <- finalize_and_predict(tree_spec, fredmd_recipe, tree_results, treated_data, holdout_data, seed = 5)
rm(tree_spec, tree_workflow, tree_results)

# 7. Random Forest Forecast
load("data/tuned_rf.RData")
pred_rf <- finalize_and_predict(rf_spec, fredmd_recipe, rf_results, treated_data, holdout_data, seed = 6)
rm(rf_spec, rf_workflow, rf_results)

# 8. Boosted Trees (XGBoost) Forecast
load("data/tuned_xgb.RData")
pred_xgb <- finalize_and_predict(xgb_spec, fredmd_recipe, xgb_results, treated_data, holdout_data, seed = 7)
rm(xgb_spec, xgb_workflow, xgb_results)

# 9. Multilayer Perceptron (MLP) Forecast
load("data/tuned_mlps.RData")
# Pick the best specification structure based on validation RMSE
rmse_1 <- show_best(mlp_1_results, metric = "rmse", n = 1)$mean
rmse_2 <- show_best(mlp_2_results, metric = "rmse", n = 1)$mean

if (rmse_1 < rmse_2) {
  pred_mlp <- finalize_and_predict(mlp_1_spec, mlp_recipe, mlp_1_results, treated_data, holdout_data, seed = 8)
} else {
  pred_mlp <- finalize_and_predict(mlp_2_spec, mlp_recipe, mlp_2_results, treated_data, holdout_data, seed = 8)
}
rm(mlp_1_spec, mlp_1_workflow, mlp_1_results,
   mlp_2_spec, mlp_2_workflow, mlp_2_results,
   mlp_recipe)

# Compile all predictions
pred_compiled <- holdout_data %>%
  select(date, actual = CPIAUCSL) %>%
  mutate(
    pred_ar1   = pred_ar1,
    pred_rw    = pred_rw,
    pred_lasso = pred_lasso,
    pred_ridge = pred_ridge,
    pred_elnet = pred_elnet,
    pred_tree  = pred_tree,
    pred_rf    = pred_rf,
    pred_xgb   = pred_xgb,
    pred_mlp   = pred_mlp
  )

# ----- Tab 2: Accuracy Tests -----
message("Running Diebold-Mariano tests...")

# Define competing models and baseline references
competing_models <- c("lasso", "ridge", "elnet", "tree", "rf", "xgb", "mlp")
baselines <- c("ar1", "rw")

# Reshape predictions and compute out-of-sample forecast errors
errors <- pred_compiled %>%
  pivot_longer(
    cols = starts_with("pred_"),
    names_to = "model",
    names_prefix = "pred_",
    values_to = ".pred"
  ) %>%
  mutate(error = actual - .pred) %>%
  select(date, model, error)

# Run Diebold-Mariano tests across all model-baseline pairs
dm_results <- crossing(
  model = competing_models,
  baseline = baselines
) %>%
  mutate(
    p_value = map2_dbl(model, baseline, function(m, b) {
      err_m <- errors %>% filter(model == m) %>% pull(error)
      err_b <- errors %>% filter(model == b) %>% pull(error)
      
      # Test whether the ML model forecast error is significantly less than the baseline
      dm <- dm.test(err_m, err_b, alternative = "less", h = 1)
      dm$p.value
    })
  )

# Filter and display only the models where the baseline null hypothesis is rejected (p < 0.05)
significant_models <- dm_results %>%
  group_by(model) %>%
  filter(p_value < 0.05) %>%
  ungroup() %>% 
  arrange(baseline, p_value)

print(significant_models)

# ----- Tab 3: Save Results -----
message("Saving holdout predictions and test results...")
save(pred_compiled, dm_results, file = "data/holdout_results.RData")

# ------------------------------------------------------------------------------
# 2.3 More on Evaluation Metrics & Visualizations
# ------------------------------------------------------------------------------

# ----- Tab 1: RMSE and MAE Comparison -----
message("Generating RMSE/MAE error metric comparisons...")

# Reshape the predictions into long format
pred_long <- pred_compiled %>%
  pivot_longer(
    cols = starts_with("pred_"),
    names_to = "model",
    names_prefix = "pred_",
    values_to = ".pred"
  )

# Calculate RMSE and MAE grouped by model
metrics_summary <- pred_long %>%
  group_by(model) %>%
  metrics(truth = actual, estimate = .pred) %>%
  filter(.metric %in% c("rmse", "mae"))

# Plot performance metrics across all models
p_metrics <- metrics_summary %>%
  mutate(.metric = toupper(.metric)) %>%
  ggplot(aes(x = reorder(model, .estimate), y = .estimate, fill = model)) +
  geom_col(show.legend = FALSE, alpha = 0.8) +
  facet_wrap(~.metric, scales = "free_y") +
  theme_minimal() +
  labs(
    title = "Out-of-Sample Accuracy Metrics",
    x = "Model",
    y = "Error Metric Value"
  ) +
  coord_flip()

print(p_metrics)

# ----- Tab 2: Forecast vs. Actual -----
message("Generating Forecast vs. Actual comparison plot...")

# Select the top 3 ML models by RMSE (excluding baseline benchmarks)
top_ml_models <- metrics_summary %>%
  filter(.metric == "rmse", !model %in% c("ar1", "rw")) %>%
  arrange(.estimate) %>%
  slice_head(n = 3) %>%
  pull(model)

# Plot predicted vs actual time series for the baselines and top ML models
p_forecasts <- pred_compiled %>%
  pivot_longer(
    cols = all_of(c("actual", "pred_ar1", "pred_rw", paste0("pred_", top_ml_models))),
    names_to = "series",
    values_to = "value"
  ) %>%
  mutate(
    series = case_when(
      series == "actual" ~ "Actual CPI Inflation",
      series == "pred_ar1" ~ "AR(1) Baseline",
      series == "pred_rw" ~ "Random Walk (RW) Baseline",
      TRUE ~ paste0("Model: ", str_to_upper(gsub("pred_", "", series)))
    )
  ) %>%
  ggplot(aes(x = date, y = value, color = series, linetype = series)) +
  geom_line(linewidth = 0.8) +
  theme_minimal() +
  labs(
    title = "Actual Inflation vs. Top Model Forecasts",
    subtitle = "Comparing the AR(1) and RW baselines against the top 3 machine learning configurations",
    x = "Date",
    y = "Inflation Rate",
    color = "Model/Series",
    linetype = "Model/Series"
  ) +
  theme(legend.position = "bottom")

print(p_forecasts)

# ----- Tab 3: Directional Accuracy -----
message("Generating directional (sign) accuracy comparison plot...")

# Calculate directional accuracy for all models
directional_accuracy_all <- pred_compiled %>%
  pivot_longer(
    cols = starts_with("pred_"),
    names_to = "model",
    names_prefix = "pred_",
    values_to = ".pred"
  ) %>%
  group_by(model) %>%
  mutate(
    actual_dir = sign(actual - lag(actual, 1)),
    pred_dir   = sign(.pred - lag(actual, 1))
  ) %>%
  drop_na() %>%
  summarise(accuracy = mean(actual_dir == pred_dir) * 100)

# Plot directional accuracy comparison
p_dir <- ggplot(directional_accuracy_all, aes(x = reorder(model, accuracy), y = accuracy, fill = model)) +
  geom_col(show.legend = FALSE, alpha = 0.8) +
  geom_hline(yintercept = 50, linetype = "dashed", color = "red", linewidth = 0.8) +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  theme_minimal() +
  labs(
    title = "Directional Accuracy Comparison",
    subtitle = "Percentage of correct direction-of-change forecasts (Red line = 50% baseline)",
    x = "Model",
    y = "Accuracy Rate (%)"
  ) +
  coord_flip()

print(p_dir)

# ----- Tab 4: Cumulative Forecast Errors -----
message("Generating cumulative forecast errors (SSE) plot...")

# Compute cumulative sum of squared forecast errors
cum_errors <- pred_compiled %>%
  pivot_longer(
    cols = starts_with("pred_"),
    names_to = "model",
    names_prefix = "pred_",
    values_to = ".pred"
  ) %>%
  mutate(sq_error = (actual - .pred)^2) %>%
  group_by(model) %>%
  arrange(date) %>%
  mutate(cum_sse = cumsum(sq_error)) %>%
  ungroup()

# Plot cumulative errors over the holdout test period
p_cum <- ggplot(cum_errors, aes(x = date, y = cum_sse, color = model)) +
  geom_line(linewidth = 0.8) +
  theme_minimal() +
  labs(
    title = "Cumulative Sum of Squared Forecast Errors (SSE)",
    subtitle = "Lower curves indicate better out-of-sample performance over time",
    x = "Date",
    y = "Cumulative SSE",
    color = "Model"
  )

print(p_cum)
