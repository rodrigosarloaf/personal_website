# ==============================================================================
# SIdE Summer Course 2026
# Lecture 2: Regularized Linear Models & Covariance Shrinkage
# Script: Part 2 - Linear and Non-linear Covariance Shrinkage
# ==============================================================================

# Load required libraries
library(tidyverse)
library(cvCovEst)

# ------------------------------------------------------------------------------
# 2.2 Covariance Estimation using `{cvCovEst}`
# ------------------------------------------------------------------------------

# Load daily prices from Lecture 1
message("Loading stock prices data...")
prices <- read_csv("data/djia_prices.csv")

# Confirm there are no missing values
# Safety check before calling drop_na() later
anyNA(prices)

# Aggregate to monthly returns
message("Aggregating daily prices to monthly returns...")
monthly_returns <- prices %>%
  mutate(month = floor_date(date, "month")) %>%
  group_by(symbol, month) %>%
  filter(date == max(date)) %>%
  ungroup() %>%
  group_by(symbol) %>%
  arrange(date, .by_group = TRUE) %>%
  mutate(returns = adjusted_close / lag(adjusted_close) - 1) %>%
  drop_na(returns) %>%
  select(symbol, date = month, returns)

# Pivot returns to a wide matrix format
returns_wide <- monthly_returns %>%
  pivot_wider(names_from = symbol, values_from = returns) %>%
  drop_na()

dates <- returns_wide$date
returns_matrix <- returns_wide %>% select(-date) %>% as.matrix()

# Save the aggregated monthly returns to disk for future use
write_csv(monthly_returns, "data/monthly_returns.csv")

# Sample Covariance Matrix
S <- cov(returns_matrix)

# Ledoit-Wolf Linear Shrinkage
cov_linear <- linearShrinkLWEst(returns_matrix)

# Ledoit-Wolf Non-linear Shrinkage
cov_nonlinear <- nlShrinkLWEst(returns_matrix)

# Correlation Heatmap Comparison
# Helper function to convert covariance matrix to long format correlation data
# (We explicitly assign row and column names because some estimators, like nlShrinkLWEst, drop them)
get_cor_long <- function(cov_matrix, estimator_name) {
  colnames(cov_matrix) <- colnames(returns_matrix)
  rownames(cov_matrix) <- colnames(returns_matrix)
  
  cov2cor(cov_matrix) %>%
    as.data.frame() %>%
    rownames_to_column(var = "Asset1") %>%
    pivot_longer(-Asset1, names_to = "Asset2", values_to = "Correlation") %>%
    mutate(Estimator = estimator_name)
}

# Combine all three estimators
all_cor_long <- bind_rows(
  get_cor_long(S, "1. Sample Covariance"),
  get_cor_long(cov_linear, "2. Ledoit-Wolf Linear"),
  get_cor_long(cov_nonlinear, "3. Ledoit-Wolf Non-linear")
)

# Plot the correlation heatmaps side-by-side
cor_heatmap <- all_cor_long %>%
  ggplot(aes(x = Asset1, y = Asset2, fill = Correlation)) +
  geom_tile() +
  scale_fill_gradient2(
    low = "blue", 
    mid = "white", 
    high = "red", 
    limits = c(-1, 1)
  ) +
  facet_wrap(~ Estimator, ncol = 3) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 6),
    axis.text.y = element_text(size = 6),
    strip.text = element_text(face = "bold", size = 10),
    legend.position = "bottom"
  ) +
  labs(
    title = "Asset Correlation Heatmap Comparison",
    subtitle = "Comparing Sample Covariance vs. Ledoit-Wolf Linear and Non-linear Shrinkage",
    x = "",
    y = "",
    fill = "Correlation"
  )

print(cor_heatmap)

# ------------------------------------------------------------------------------
# 2.3 Global Minimum Variance Portfolio (GMVP)
# ------------------------------------------------------------------------------

# Function to compute GMVP weights
compute_gmvp <- function(sigma) {
  ones <- rep(1, ncol(sigma))
  sigma_inv <- MASS::ginv(sigma) # Using the pseudo-inverse if the provided matrix is singular 
  weights <- (sigma_inv %*% ones) / as.numeric(t(ones) %*% sigma_inv %*% ones)
  return(as.vector(weights))
}

# Compute weights
w_sample <- compute_gmvp(S)
w_linear <- compute_gmvp(cov_linear)
w_nonlinear <- compute_gmvp(cov_nonlinear)

# Pivot weights to long format for plotting
weights_long <- tibble(
  Asset = colnames(returns_matrix),
  `1. Sample Covariance` = w_sample,
  `2. Ledoit-Wolf Linear` = w_linear,
  `3. Ledoit-Wolf Non-linear` = w_nonlinear
) %>%
  pivot_longer(-Asset, names_to = "Estimator", values_to = "Weight")

# Plot weights allocation comparison
weights_plot <- weights_long %>%
  ggplot(aes(x = Asset, y = Weight, fill = Estimator)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  facet_wrap(~ Estimator, ncol = 3) +
  theme_minimal() +
  theme(
    axis.text.y = element_text(size = 7),
    axis.text.x = element_text(size = 8),
    strip.text = element_text(face = "bold", size = 10)
  ) +
  labs(
    title = "GMVP Portfolio Allocations Comparison",
    subtitle = "Comparing Asset Weights across Covariance Estimators (Ordered by Sample Weights)",
    x = "Asset",
    y = "Portfolio Weight"
  )

print(weights_plot)

# ------------------------------------------------------------------------------
# 2.4 Out-of-Sample Portfolio Backtesting
# ------------------------------------------------------------------------------

window_size <- 24
n_forecasts <- nrow(returns_matrix) - window_size

portfolio_returns <- tibble(
  Date = dates[(window_size + 1):length(dates)],
  Sample = numeric(n_forecasts),
  Linear = numeric(n_forecasts),
  Nonlinear = numeric(n_forecasts)
)

message("Running rolling window backtest...")
for (i in 1:n_forecasts) {
  train_idx <- i:(i + window_size - 1)
  test_idx <- i + window_size
  
  X_train <- returns_matrix[train_idx, ]
  r_test <- returns_matrix[test_idx, ]
  
  # Estimations
  S_roll <- cov(X_train)
  cov_linear_roll <- linearShrinkLWEst(X_train)
  cov_nonlinear_roll <- nlShrinkLWEst(X_train)
  
  # GMVP Weights
  w_s <- compute_gmvp(S_roll)
  w_l <- compute_gmvp(cov_linear_roll)
  w_nl <- compute_gmvp(cov_nonlinear_roll)
  
  # Realized returns
  portfolio_returns$Sample[i] <- sum(w_s * r_test)
  portfolio_returns$Linear[i] <- sum(w_l * r_test)
  portfolio_returns$Nonlinear[i] <- sum(w_nl * r_test)
}

# Evaluate annualized volatility (risk)
portfolio_risk <- portfolio_returns %>%
  pivot_longer(-Date, names_to = "Estimator", values_to = "Return") %>%
  group_by(Estimator) %>%
  summarize(
    Mean_Return = mean(Return) * 12,
    OOS_Volatility = sd(Return) * sqrt(12)
  )

print(portfolio_risk)

# Plot realized cumulative returns
cum_returns_plot <- portfolio_returns %>%
  pivot_longer(-Date, names_to = "Estimator", values_to = "Return") %>%
  group_by(Estimator) %>%
  mutate(Cumulative_Return = cumprod(1 + Return) - 1) %>%
  ggplot(aes(x = Date, y = Cumulative_Return, color = Estimator)) +
  geom_line(linewidth = 1) +
  theme_minimal() +
  labs(
    title = "Out-of-Sample GMVP Performance",
    subtitle = "Comparing Covariance Estimators: Sample vs. Ledoit-Wolf Shrinkage",
    x = "Date",
    y = "Cumulative Return",
    color = "Estimator"
  )

print(cum_returns_plot)
