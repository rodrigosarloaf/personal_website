# ==============================================================================
# SIdE Summer Course 2026
# Lecture 1: Data Ecosystems, Tidymodels & Tidyfinance
# Script: Part 1 - FRED-MD Database & Tidymodels Workflow
# ==============================================================================

# Load required libraries
# Note: install cykbennie/fbi before running: remotes::install_github("cykbennie/fbi")
library(tidyverse)
library(fbi)
library(tidymodels)

# ------------------------------------------------------------------------------
# 1.1 The FRED-MD Dataset & Stationarity Transformations
# ------------------------------------------------------------------------------

# 1. Download raw (untransformed) FRED-MD series
message("Downloading FRED-MD database...")
req_fredmd <- httr::GET("https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/2026-05-md.csv")
raw_data <- httr::content(req_fredmd, as = "parsed")

# Inspect raw data
head(raw_data)

# 2. Retrieve default tcodes
tcodes <- unlist(raw_data[1, 2:ncol(raw_data)], use.names = TRUE)

# 3. Identify Group 7 (Prices) variables from the metadata
data("fredmd_description")
price_vars <- fredmd_description %>%
  filter(group == "Prices") %>%
  pull(fred)

# 4. Modify price variables tcode from 6 to 7
tcodes[(names(tcodes) %in% price_vars) & (tcodes == 6)] <- 7

# 5. Insert back the modified data
modif_data <- raw_data
modif_data[1, 2:ncol(raw_data)] <- as.list(tcodes)

# Create data directory if it doesn't exist, then write to disk
dir.create("data", showWarnings = FALSE)
write_csv(modif_data, "data/fredmd_custom.csv")

# 6. Load the modified dataset using fbi::fredmd with transform = TRUE
message("Loading and stationarizing dataset...")
macro_data <- fredmd(
  file = "data/fredmd_custom.csv",
  transform = TRUE
)

# 7. Keeping only the variables with all observations between 1960-01-01 and 2025-09-01
macro_data <- macro_data %>% 
  as_tibble() %>% 
  filter(date >= ymd("1960-01-01"), date <= ymd("2025-09-01"))

# Filtering variables with NAs
na_variables <- colSums(is.na(macro_data))
na_variables <- names(na_variables[na_variables > 0])
message("Variables containing NAs:")
print(na_variables)

# Keep only balanced panel
balanced_data <- macro_data %>% select(-any_of(na_variables))
write_csv(balanced_data, "data/balanced_fredmd.csv")

# Final dimensions of the cleaned dataset
message("Balanced dataset dimensions:")
print(dim(balanced_data))


# ------------------------------------------------------------------------------
# 1.2 Modeling in the Tidymodels Environment
# ------------------------------------------------------------------------------

# 0. Offline preprocessing steps
# Some operations, like generating lag structures, should be performed offline (outside {recipes}).
# Following Medeiros et al. (2021), we generate 4 lags for all variables, create a dummy, select only targets and lagged predictors, and drop the first few rows.
treated_data <- balanced_data %>%
  mutate(across(2:last_col(), # All columns except date
                list(lag1 = ~dplyr::lag(.x, n = 1),
                     lag2 = ~dplyr::lag(.x, n = 2),
                     lag3 = ~dplyr::lag(.x, n = 3),
                     lag4 = ~dplyr::lag(.x, n = 4)), 
                .names = "{.fn}_{.col}"),
         d200811 = ifelse(date == ymd("2008-11-01"), 1, 0)
         ) %>%
  # Keep only the target variable, the dummy, and the lagged predictors
  select(date, CPIAUCSL, d200811, starts_with("lag")) %>%
  # Skip the first 4 rows containing NAs from lagging
  filter(date >= ymd("1960-05-01"))

write_csv(treated_data, "data/treated_fredmd.csv")

# Finally, let's create a hold out the last 48 months for final model comparison
n_test <- 48
train_val_data <- treated_data %>% slice(1:(n() - n_test))
holdout_test_data <- treated_data %>% slice_tail(n = n_test)

# Save both datasets to disk
write_csv(train_val_data, "data/train_val_fredmd.csv")
write_csv(holdout_test_data, "data/test_fredmd.csv")


# 1. Splitting the data with {rsample}
message("Setting up time series splits...")
folds <- rolling_origin(
  train_val_data,
  initial = 360,    # 30 years of training data
  assess = 1,       # 1 month of evaluation data
  cumulative = FALSE # FALSE for rolling window
)

# Inspect the structure of the folds dataframe
head(folds)
message("Dimensions of folds:")
print(dim(folds))

# Inspect a specific split
first_split <- folds$splits[[1]]
print(first_split) # Shows <Analysis/Assessment/Total> sizes

# Extract training and testing data for this fold
train_data <- analysis(first_split)
test_data <- assessment(first_split)

head(train_data)
head(test_data)


# 2. Preprocessing with {recipes}
# Finishing the preprocessing pipeline following Medeiros et al. (2021)
message("Building preprocessing recipe...")
fredmd_recipe <- recipe(CPIAUCSL ~ ., data = train_val_data) %>%
  # Drop the date variable so it is not used in models
  step_rm(date) %>%
  # Apply PCA per lag-set
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
  # Drop the dummy if it contains all zeros
  step_zv(d200811)

# Inspect recipe steps
print(fredmd_recipe)

# We can call prep() and bake() to evaluate the recipe on training data and get the transformed dataset
prepped_recipe <- prep(fredmd_recipe, training = train_data)
transformed_train_data <- bake(prepped_recipe, new_data = train_data)

message("Transformed training data:")
head(transformed_train_data)
dim(transformed_train_data)

# Apply the same preprocessing recipe to the test data to get the transformed test set
transformed_test_data <- bake(prepped_recipe, new_data = test_data)
head(transformed_test_data)
