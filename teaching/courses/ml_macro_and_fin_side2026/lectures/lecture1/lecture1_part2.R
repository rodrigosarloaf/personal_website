# ==============================================================================
# SIdE Summer Course 2026
# Lecture 1: Data Ecosystems, Tidymodels & Tidyfinance
# Script: Part 2 - Overview of the Tidyfinance Package
# ==============================================================================

# Load required libraries
library(tidyverse)
library(tidyfinance)
library(broom)

# ------------------------------------------------------------------------------
# 2.1 Fetching Financial & Macro-Finance Data
# ------------------------------------------------------------------------------

# 1. Fetching specific individual stock prices from Yahoo Finance
message("Downloading AAPL prices...")
prices_apple <- download_data(
  domain = "stock_prices",
  symbols = "AAPL",
  start_date = "2020-01-01",
  end_date = "2025-12-31"
)
head(prices_apple)

# 2. Fetch the constituents list of the Dow Jones Industrial Average (DJIA)
message("Downloading Dow Jones constituents list...")
dow_constituents <- download_data(
  domain = "constituents",
  index = "Dow Jones Industrial Average"
)
head(dow_constituents)

# 3. Pull tickers and download prices for all DJIA components
message("Downloading prices for all Dow Jones stocks (Yahoo Finance)...")
dow_tickers <- dow_constituents %>% pull(symbol)

prices <- download_data(
  domain = "stock_prices",
  symbols = dow_tickers,
  start_date = "2020-01-01",
  end_date = "2025-12-31"
)
message("DJIA prices dataset dimensions:")
print(dim(prices))

# 4. Fetch the Fama/French 3-Factor Model data (monthly frequency)
message("Downloading Fama-French 3-factor monthly data...")
factors_ff3 <- download_data(
  domain = "famafrench",
  dataset = "factors_ff_3_monthly",
  start_date = "2020-01-01",
  end_date = "2025-12-31"
)
head(factors_ff3)

# 5. Fetch macroeconomic indicators from FRED (Inflation via CPIAUCSL)
message("Downloading Inflation (CPIAUCSL) from FRED...")
inflation <- download_data(
  domain = "fred",
  series = "CPIAUCSL",
  start_date = "2020-01-01",
  end_date = "2025-12-31"
)
head(inflation)

# 6. Save the downloaded DJIA prices and Fama-French factors to disk
message("Saving raw datasets to disk...")
dir.create("data", showWarnings = FALSE)
write_csv(prices, "data/djia_prices.csv")
write_csv(factors_ff3, "data/ff_factors.csv")


# ------------------------------------------------------------------------------
# 2.2 Returns, Aggregation & Fama-French Estimation
# ------------------------------------------------------------------------------

# 1. Aggregate daily stock prices to monthly frequency (taking last trading day of the month)
message("Aggregating daily prices to monthly returns...")
monthly_returns <- prices %>%
  mutate(month = floor_date(date, "month")) %>%
  group_by(symbol, month) %>%
  filter(date == max(date)) %>% # Keep only the last trading day of the month
  ungroup() %>%
  group_by(symbol) %>%
  arrange(date, .by_group = TRUE) %>%
  mutate(returns = adjusted_close / lag(adjusted_close) - 1) %>%
  # Drops the returns from the first month only - check with "anyNA(prices)" - we didn't have NAs before calling "lag()"
  drop_na(returns) %>% 
  select(symbol, date = month, returns)

# Inspect monthly returns
head(monthly_returns)

# 2. Align stock returns with Fama-French monthly factors and calculate excess returns
message("Aligning returns with Fama-French risk factors...")
regression_data <- monthly_returns %>%
  inner_join(factors_ff3, by = "date") %>%
  mutate(excess_return = returns - risk_free) # Excess returns (R_i - R_f)

head(regression_data)

# 3. Example: Run Fama-French 3-Factor regression for a single stock (e.g., AAPL)
message("Estimating Fama-French 3-Factor model for AAPL...")
aapl_fit <- regression_data %>%
  filter(symbol == "AAPL") %>%
  lm(excess_return ~ mkt_excess + smb + hml, data = .)

print(summary(aapl_fit))

# 4. Programmatically run the regression for all DJIA components using purrr and broom
message("Estimating models for all DJIA components programmatically...")
ff_results <- regression_data %>%
  group_by(symbol) %>%
  # Creates a column of tibbles for the tickers 
  nest() %>%
  mutate(
    # Applies "lm()" inside the column of tibbles
    model = map(data, ~ lm(excess_return ~ mkt_excess + smb + hml, data = .x)),
    coefficients = map(model, tidy)
  ) %>%
  unnest(coefficients) %>%
  select(symbol, term, estimate, std.error, statistic, p.value)

# Inspect the coefficients (alphas and factor loadings/betas) for all stocks
message("First few Fama-French estimates:")
print(head(ff_results))

# 5. Visualize the estimated factor loadings with 95% confidence intervals
# (We exclude the intercept/alpha for focus on the risk exposures)
message("Generating Fama-French factor loading plots...")
ff_plot <- ff_results %>%
  filter(term != "(Intercept)") %>%
  ggplot(aes(x = estimate, y = reorder(symbol, estimate), color = term)) +
  geom_point() +
  geom_errorbarh(aes(xmin = estimate - 1.96 * std.error, xmax = estimate + 1.96 * std.error), height = 0.2) +
  # Creates separate panels for each factor
  facet_wrap(~term, scales = "free_x") +
  theme_minimal() +
  labs(
    title = "Fama-French 3-Factor Loadings for DJIA Stocks",
    subtitle = "Point estimates with 95% confidence intervals (2020-2025)",
    x = "Beta Coefficient Estimate",
    y = "Stock Symbol",
    color = "Factor"
  ) +
  theme(legend.position = "none")

# Display the plot in the session
print(ff_plot)

