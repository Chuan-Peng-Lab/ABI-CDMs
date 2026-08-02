# ==============================================================================
# Script: run_models.R
# Purpose: Data preprocessing and Bayesian model fitting for cognitive factors.
#          Separates heavy computation from downstream analysis/visualization.
# ==============================================================================

# 1. Setup and Package Loading -------------------------------------------------
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  tidyverse, # Data manipulation
  brms,      # Bayesian modeling
  psych      # For initial test-retest ICC calculation
)
source_path <- tryCatch(sys.frame(1)$ofile, error = function(error) NULL)
script_dir <- if (is.null(source_path)) {
  normalizePath("scripts/07_parameter_analysis", mustWork = TRUE)
} else {
  normalizePath(dirname(source_path), mustWork = TRUE)
}
repo_root <- normalizePath(file.path(script_dir, "..", ".."), mustWork = TRUE)
intermediate_dir <- file.path(repo_root, "results", "intermediate")
dir.create(intermediate_dir, recursive = TRUE, showWarnings = FALSE)

# Global Settings
GLOBAL_SEED <- 20260218
set.seed(GLOBAL_SEED)
CORES <- 4
THREADS_PER_CORE <- 16 # Adjust based on your CPU, or remove threading() in models if using rstan

# Note: If you do not have cmdstanr installed, remove the `backend = "cmdstanr"`
# line from the model functions below, or install it via:
# library(cmdstanr); install_cmdstan()


# 2. Data Preprocessing Functions ----------------------------------------------

#' Prepare Data for Analysis (Wide to Long & Z-score)
prep_analysis_data <- function(data,
                               factor_cols = c("Processing Efficiency", "Decision Caution",
                                               "Non-decision time", "Inhibitory process"),
                               id_cols = c("subject_id", "task_id", "author_year", "task_name")) {

  long_data <- data %>%
    pivot_longer(cols = all_of(factor_cols),
                 names_to = "Factor_Type",
                 values_to = "Raw_Value") %>%
    mutate(Factor_Type = as.factor(Factor_Type)) %>%
    # Standardize (Z-score) within each Factor_Type
    group_by(Factor_Type) %>%
    mutate(Value = as.numeric(scale(Raw_Value))) %>%
    ungroup()

  return(long_data)
}

#' Preprocess Data for Cross-Task Bayesian Analysis
preprocess_cognitive_data <- function(data) {
  data %>%
    # Create Unique Subject ID (Lab + ID)
    mutate(unique_subj_id = paste(author_year, subject_id, sep = "_")) %>%
    # Standardize again after combining (safety measure)
    group_by(Factor_Type) %>%
    mutate(Value_Scaled = as.numeric(scale(Value))) %>%
    ungroup() %>%
    # Ensure variables are factors
    mutate(Factor_Type = as.factor(Factor_Type),
           unique_subj_id = as.factor(unique_subj_id),
           author_year = as.factor(author_year))
}

#' Calculate Test-Retest ICC for Each Subgroup
calculate_subgroup_iccs <- function(data) {

  get_psych_icc <- function(d) {
    d_wide <- d %>%
      select(subject_id, session_id, Value) %>%
      pivot_wider(names_from = session_id, values_from = Value) %>%
      select(-subject_id)

    if(ncol(d_wide) < 2 || nrow(d_wide) < 5) return(NA)

    tryCatch({
      res <- psych::ICC(d_wide, missing = FALSE, lmer = FALSE)
      return(res$results["Single_fixed_raters", "ICC"])
    }, error = function(e) return(NA))
  }

  data %>%
    group_by(author_year, task_name, Factor_Type) %>%
    nest() %>%
    mutate(ICC = map_dbl(data, get_psych_icc)) %>%
    select(-data) %>%
    ungroup() %>%
    filter(!is.na(ICC)) %>%
    # Clamp to avoid Inf in Fisher Z, then apply Fisher Z Transformation
    mutate(
      ICC = pmin(pmax(ICC, -0.99), 0.99),
      ICC_FisherZ = 0.5 * log((1 + ICC) / (1 - ICC))
    )
}


# 2b. Analysis Helper Functions ------------------------------------------------

#' Format Author and Task Labels for Display
#' "clayson2025" -> "Clayson 2025"; "flanker" -> "Flanker".
format_data_labels <- function(data) {
  data %>%
    mutate(
      author_year = str_replace(author_year, "^([a-zA-Z]+)([0-9]+)$", "\\1 \\2"),
      author_year = str_to_title(author_year),
      task        = str_to_title(task_name)
    )
}

#' Calculate Group Summary Statistics (Mean, SE, 95% CI)
calculate_summary_stats <- function(data, group_vars, value_var = "Value") {
  data %>%
    group_by(across(all_of(group_vars))) %>%
    summarise(
      mean_val = mean(.data[[value_var]], na.rm = TRUE),
      sd_val   = sd(.data[[value_var]],   na.rm = TRUE),
      n        = n(),
      .groups  = "drop"
    ) %>%
    mutate(
      se_val   = sd_val / sqrt(n),
      ci_lower = mean_val - 1.96 * se_val,
      ci_upper = mean_val + 1.96 * se_val
    )
}

#' Extract Posterior ICC Draws from Fitted brms Model
#' Returns: .draw, Factor_Type, ICC_conditional, ICC_marginal
get_posterior_icc <- function(model) {
  draws <- as_draws_df(model)

  clean_factor_names <- function(data) {
    data %>% mutate(Factor_Type = case_when(
      str_detect(Factor_Raw, "Controlprocess")    ~ "Control",
      str_detect(Factor_Raw, "DecisionCaution")   ~ "Decision_Caution",
      str_detect(Factor_Raw, "Inhibitoryprocess") ~ "Inhibitory",
      str_detect(Factor_Raw, "NonMdecisiontime")  ~ "Non_decision_time",
      TRUE ~ Factor_Raw
    ))
  }

  post_tau_subj <- draws %>%
    select(.draw, starts_with("sd_unique_subj_id__Factor_Type")) %>%
    pivot_longer(-.draw, names_to = "Factor_Raw", values_to = "tau_subj") %>%
    mutate(Factor_Raw = str_remove(Factor_Raw, "sd_unique_subj_id__Factor_Type")) %>%
    clean_factor_names()

  post_tau_lab <- draws %>%
    select(.draw, starts_with("sd_author_year__Factor_Type")) %>%
    pivot_longer(-.draw, names_to = "Factor_Raw", values_to = "tau_lab") %>%
    mutate(Factor_Raw = str_remove(Factor_Raw, "sd_author_year__Factor_Type")) %>%
    clean_factor_names()

  post_sigma <- draws %>%
    select(.draw, starts_with("b_sigma_Factor_Type")) %>%
    pivot_longer(-.draw, names_to = "var_name", values_to = "log_sigma") %>%
    mutate(
      Factor_Raw  = str_extract(var_name, "(?<=b_sigma_Factor_Type)[^:]+"),
      sigma_resid = exp(log_sigma)
    ) %>%
    clean_factor_names() %>%
    group_by(.draw, Factor_Type) %>%
    summarize(mean_var_resid = mean(sigma_resid^2), .groups = "drop")

  post_tau_subj %>%
    left_join(post_tau_lab, by = c(".draw", "Factor_Type")) %>%
    left_join(post_sigma,   by = c(".draw", "Factor_Type")) %>%
    mutate(
      var_subj        = tau_subj^2,
      var_lab         = tau_lab^2,
      ICC_conditional = var_subj / (var_subj + mean_var_resid),
      ICC_marginal    = var_subj / (var_subj + var_lab + mean_var_resid)
    ) %>%
    select(.draw, Factor_Type, ICC_conditional, ICC_marginal)
}

#' Calculate All Pairwise Posterior Differences Between Factor ICCs
#' Returns: Comparison, Mean_Difference, Prob_Direction, Significant_95
calculate_pairwise_stats <- function(data, icc_col = "ICC") {
  factors <- unique(data$Factor_Type)
  combs   <- combn(factors, 2, simplify = FALSE)
  map_df(combs, function(pair) {
    p1        <- data %>% filter(Factor_Type == pair[1]) %>% pull(!!sym(icc_col))
    p2        <- data %>% filter(Factor_Type == pair[2]) %>% pull(!!sym(icc_col))
    diff_dist <- p1 - p2
    tibble(
      Comparison      = paste0(pair[1], " - ", pair[2]),
      Mean_Difference = mean(diff_dist),
      Prob_Direction  = max(mean(diff_dist > 0), mean(diff_dist < 0)),
      Significant_95  = !(quantile(diff_dist, 0.025) < 0 & quantile(diff_dist, 0.975) > 0)
    )
  })
}


# 3. Model Fitting Functions ---------------------------------------------------

#' Model 1: Cross-Task Consistency (Distributional Model)
#'
#' @param data Processed long-format dataframe from preprocess_cognitive_data().
#' @param cores Number of CPU cores to use for parallel chains.
#' @param seed Random seed for reproducibility.
#' @param iter Number of iterations per chain (default: 5000).
#' @param threads_per_core Threads per core for cmdstanr threading (default: THREADS_PER_CORE).
fit_icc_model <- function(data, cores, seed, iter = 5000,
                          threads_per_core = THREADS_PER_CORE) {
  f <- bf(
    Value_Scaled ~ 0 + Factor_Type:task_name +
      (0 + Factor_Type | author_year) +
      (0 + Factor_Type | unique_subj_id),
    sigma ~ 0 + Factor_Type:task_name
  )

  brm(
    formula = f,
    data = data,
    family = student(),
    chains = 4,
    cores = cores,
    threads = threading(threads_per_core),
    iter = iter,
    seed = seed,
    control = list(adapt_delta = 0.95),
    backend = "cmdstanr",
    file = file.path(intermediate_dir, "cross_task_icc_model")
  )
}

#' Model 2: Temporal Stability (Random Effects for Task and Lab)
#'
#' @param icc_data Result from calculate_subgroup_iccs().
#' @param cores Number of CPU cores.
#' @param seed Random seed.
#' @param threads_per_core Threads per core for cmdstanr threading.
fit_reliability_meta_model <- function(icc_data, cores, seed,
                                       threads_per_core = THREADS_PER_CORE) {
  f <- bf(ICC_FisherZ ~ 0 + Factor_Type + (1 | task_name) + (1 | author_year))

  priors <- c(
    set_prior("normal(0, 1)", class = "b"),
    set_prior("cauchy(0, 0.5)", class = "sd")
  )

  brm(
    formula = f,
    data = icc_data,
    family = gaussian(),
    prior = priors,
    cores = cores,
    chains = 4,
    iter = 5000,
    threads = threading(threads_per_core),
    control = list(adapt_delta = 0.98),
    seed = seed,
    backend = "cmdstanr",
    file = file.path(intermediate_dir, "reliability_meta_model")
  )
}

#' Model 3: Task-Specific Temporal Stability (Fixed Effect for Task)
#'
#' @param icc_data Result from calculate_subgroup_iccs().
#' @param cores Number of CPU cores.
#' @param seed Random seed.
#' @param threads_per_core Threads per core for cmdstanr threading.
fit_task_fixed_meta_model <- function(icc_data, cores, seed,
                                      threads_per_core = THREADS_PER_CORE) {
  f <- bf(ICC_FisherZ ~ 0 + task_name + (1 | author_year) + (1 | Factor_Type))

  priors <- c(
    set_prior("normal(0, 1)", class = "b"),
    set_prior("cauchy(0, 0.5)", class = "sd")
  )

  brm(
    formula = f,
    data = icc_data,
    family = gaussian(),
    prior = priors,
    cores = cores,
    chains = 4,
    iter = 5000,
    threads = threading(threads_per_core),
    control = list(adapt_delta = 0.98),
    seed = seed,
    backend = "cmdstanr",
    file = file.path(intermediate_dir, "task_reliability_meta_model")
  )
}


# 4. Execution Pipeline --------------------------------------------------------
# Guard: Only run the pipeline when this script is executed directly (e.g., via Rscript).
# When sourced from analyze_factor_reliability.Rmd, only the functions above are loaded.
if (sys.nframe() == 0) {

cat("Starting Execution Pipeline...\n")

# --- Dataset 1: Cross-Task Data ---
cat("Loading and processing cross-task data...\n")
df_cross <- read.csv(file.path(intermediate_dir, "factor_scores.csv"), check.names = FALSE)
selected_cross <- df_cross %>% select(subject_id, task_id, author_year, task_name,
                                      `Processing Efficiency`, `Decision Caution`,
                                      `Non-decision time`, `Inhibitory process`)

df_long_cross <- prep_analysis_data(selected_cross)
df_long_clean <- preprocess_cognitive_data(df_long_cross)

# --- Dataset 2: Temporal (Test-Retest) Data ---
cat("Loading and processing temporal test-retest data...\n")
df_retest <- read.csv(file.path(intermediate_dir, "factor_scores_retest.csv"), check.names = FALSE)
selected_retest <- df_retest %>% select(subject_id, session_id, task_id, author_year, task_name,
                                        `Processing Efficiency`, `Decision Caution`,
                                        `Non-decision time`, `Inhibitory process`)

df_long_retest <- prep_analysis_data(selected_retest, id_cols = c("subject_id", "session_id", "task_id", "author_year", "task_name"))
subgroup_iccs <- calculate_subgroup_iccs(df_long_retest)


# --- Fitting Models ---
# The models will be cached to disk automatically due to the 'file' argument.
# If the file already exists, brms will just load it instead of re-running.

cat("Fitting Model 1: Cross-Task ICC Distributional Model...\n")
icc_model_fitted <- fit_icc_model(df_long_clean, cores = CORES, seed = GLOBAL_SEED)

cat("Fitting Model 2: Temporal Stability (Meta-Regression)...\n")
meta_model <- fit_reliability_meta_model(subgroup_iccs, cores = CORES, seed = GLOBAL_SEED)

cat("Fitting Model 3: Task-Specific Temporal Stability...\n")
task_meta_model <- fit_task_fixed_meta_model(subgroup_iccs, cores = CORES, seed = GLOBAL_SEED)

cat("All models fitted and saved successfully!\n")

} # end if(sys.nframe() == 0)
