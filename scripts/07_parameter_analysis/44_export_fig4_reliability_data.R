#!/usr/bin/env Rscript

# Export reliability data needed by the v8 Fig4 3x3 matplotlib compositor.
# This script does not generate figures. It loads cached brms models and writes
# compact CSV files that can be consumed by Python.

suppressPackageStartupMessages({
  if (!require("pacman", quietly = TRUE)) install.packages("pacman")
  pacman::p_load(tidyverse, brms, tidybayes, posterior, psych)
})

cmd_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", cmd_args, value = TRUE)
script_path <- if (length(file_arg) > 0) {
  sub("^--file=", "", file_arg[[1]])
} else {
  "44_export_fig4_reliability_data.R"
}
script_dir <- dirname(normalizePath(script_path))
setwd(script_dir)

prep_analysis_data <- function(data,
                               factor_cols = c("Processing Efficiency", "Decision Caution",
                                               "Non-decision time", "Inhibitory process"),
                               id_cols = c("subject_id", "task_id", "author_year", "task_name")) {
  data %>%
    pivot_longer(
      cols = all_of(factor_cols),
      names_to = "Factor_Type",
      values_to = "Raw_Value"
    ) %>%
    mutate(Factor_Type = as.factor(Factor_Type)) %>%
    group_by(Factor_Type) %>%
    mutate(Value = as.numeric(scale(Raw_Value))) %>%
    ungroup()
}

calculate_subgroup_iccs <- function(data) {
  get_psych_icc <- function(d) {
    d_wide <- d %>%
      select(subject_id, session_id, Value) %>%
      pivot_wider(names_from = session_id, values_from = Value) %>%
      select(-subject_id)

    if (ncol(d_wide) < 2 || nrow(d_wide) < 5) return(NA)

    tryCatch({
      res <- psych::ICC(d_wide, missing = FALSE, lmer = FALSE)
      res$results["Single_fixed_raters", "ICC"]
    }, error = function(e) NA)
  }

  data %>%
    group_by(author_year, task_name, Factor_Type) %>%
    nest() %>%
    mutate(ICC = map_dbl(data, get_psych_icc)) %>%
    select(-data) %>%
    ungroup() %>%
    filter(!is.na(ICC)) %>%
    mutate(
      ICC = pmin(pmax(ICC, -0.99), 0.99),
      ICC_FisherZ = 0.5 * log((1 + ICC) / (1 - ICC))
    )
}

get_posterior_icc <- function(model) {
  draws <- as_draws_df(model)

  clean_factor_names <- function(data) {
    data %>% mutate(Factor_Type = case_when(
      str_detect(Factor_Raw, "Controlprocess") ~ "Control",
      str_detect(Factor_Raw, "DecisionCaution") ~ "Decision_Caution",
      str_detect(Factor_Raw, "Inhibitoryprocess") ~ "Inhibitory",
      str_detect(Factor_Raw, "NonMdecisiontime") ~ "Non_decision_time",
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
      Factor_Raw = str_extract(var_name, "(?<=b_sigma_Factor_Type)[^:]+"),
      sigma_resid = exp(log_sigma)
    ) %>%
    clean_factor_names() %>%
    group_by(.draw, Factor_Type) %>%
    summarize(mean_var_resid = mean(sigma_resid^2), .groups = "drop")

  post_tau_subj %>%
    left_join(post_tau_lab, by = c(".draw", "Factor_Type")) %>%
    left_join(post_sigma, by = c(".draw", "Factor_Type")) %>%
    mutate(
      var_subj = tau_subj^2,
      var_lab = tau_lab^2,
      ICC_conditional = var_subj / (var_subj + mean_var_resid),
      ICC_marginal = var_subj / (var_subj + var_lab + mean_var_resid)
    ) %>%
    select(.draw, Factor_Type, ICC_conditional, ICC_marginal)
}

out_dir <- file.path(script_dir, "44_fig4_reliability_exports")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

message("Preparing test-retest ICC data...")
df_retest <- read.csv("43_subj_indices_with_EFA_scores_retest.csv", check.names = FALSE)
selected_retest <- df_retest %>%
  select(subject_id, session_id, task_id, author_year, task_name,
         `Processing Efficiency`, `Decision Caution`,
         `Non-decision time`, `Inhibitory process`)

df_long_retest <- prep_analysis_data(
  selected_retest,
  id_cols = c("subject_id", "session_id", "task_id", "author_year", "task_name")
)
subgroup_iccs <- calculate_subgroup_iccs(df_long_retest) %>%
  mutate(
    task_name = tools::toTitleCase(as.character(task_name)),
    Factor_Type = as.character(Factor_Type)
  )

message("Loading cached cross-task ICC model...")
icc_model_fitted <- readRDS("44_icc_model_fitted.rds")
icc_draws <- get_posterior_icc(icc_model_fitted) %>%
  mutate(
    ICC = ICC_conditional,
    Factor_Type = recode(
      Factor_Type,
      Control = "Processing Efficiency",
      Decision_Caution = "Decision Caution",
      Inhibitory = "Inhibitory process",
      Non_decision_time = "Non-decision time"
    )
  ) %>%
  select(.draw, Factor_Type, ICC)

message("Loading cached temporal reliability model...")
meta_model <- readRDS("44_reliability_meta_model.rds")
sd_draws <- meta_model %>%
  spread_draws(sd_author_year__Intercept, sd_task_name__Intercept) %>%
  rename(Labs = sd_author_year__Intercept, Tasks = sd_task_name__Intercept) %>%
  pivot_longer(c(Labs, Tasks), names_to = "Source", values_to = "Sigma") %>%
  mutate(Source = factor(Source, levels = c("Labs", "Tasks"))) %>%
  select(.draw, Source, Sigma)

write_csv(icc_draws, file.path(out_dir, "fig9b_cross_task_icc_draws.csv"))
write_csv(subgroup_iccs, file.path(out_dir, "fig8a_temporal_icc_subgroups.csv"))
write_csv(sd_draws, file.path(out_dir, "fig8b_temporal_sd_draws.csv"))

message("Exported reliability CSV files to: ", out_dir)
