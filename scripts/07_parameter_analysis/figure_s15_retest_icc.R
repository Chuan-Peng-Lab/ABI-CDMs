# ============================================================================
# figure_s15_retest_icc.R
# Standalone regeneration of Supplementary Figure S15
# (Test-retest model-selection consistency effect on parameter consistency, ICC)
#
# Provenance / why this script exists
#   S15 was originally produced inside 45_parameter_consistency.Rmd, chunk
#   `plot-retest-consistency` (lines 525-606), variable `p_combined_tasks`.
#   Its export line (Rmd line 603) was commented out, so the SVG was never
#   written as a standalone release artifact, and the helper `compute_retest_icc`
#   it relied on was
#   never committed. THIS script is the single canonical generator of
#   figures/supplement/figure_s15_retest_icc.svg.
#
# Reconstruction note (compute_retest_icc)
#   Sessions are NOT aligned across labs (lee2025 has s1..s19; others s1,s2 /
#   s1..s4). A naive subject x session pivot + listwise deletion would keep
#   almost no one. We therefore use the universal s1/s2 retest pair and compute
#   ICC(2,1) via psych::ICC -- mirroring compute_icc_core (Rmd lines 134-159)
#   but pivoting on session_id. This is the classic two-occasion test-retest ICC.
#
# Run: Rscript scripts/07_parameter_analysis/figure_s15_retest_icc.R
# ============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(purrr)
  library(ggplot2)
  library(patchwork)
  library(psych)
  library(latex2exp)
  library(svglite)
})

set.seed(20260323)

# --- paths -------------------------------------------------------------------
cmd_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", cmd_args, value = TRUE)
script_path <- if (length(file_arg) > 0) sub("^--file=", "", file_arg[[1]]) else "."
SCRIPT_DIR <- dirname(normalizePath(script_path))
REPO_ROOT <- normalizePath(file.path(SCRIPT_DIR, "..", ".."), mustWork = TRUE)
INTERMEDIATE_DIR <- file.path(REPO_ROOT, "results", "intermediate")
SUPPLEMENT_DIR <- file.path(REPO_ROOT, "figures", "supplement")
dir.create(SUPPLEMENT_DIR, recursive = TRUE, showWarnings = FALSE)
OUT_SVG    <- file.path(SUPPLEMENT_DIR, "figure_s15_retest_icc.svg")
OUT_PNG    <- sub("\\.svg$", ".png", OUT_SVG)

# --- parameter metadata + palette (Rmd lines 52-96) --------------------------
PARAM_CONFIG <- tibble::tribble(
  ~model, ~param_raw, ~tex_label, ~efa_factor,
  "DMC",  "$a|DMC$",        "$a$",        "Decision Caution",
  "DMC",  "$t|DMC$",        "$t$",        "Non-decision time",
  "DMC",  "$v_{c}|DMC$",    "$v_{c}$",    "Processing Efficiency",
  "DMC",  "$\\alpha|DMC$",  "$\\alpha$",  "Inhibitory process",
  "DMC",  "$\\eta|DMC$",    "$\\eta$",    "Inhibitory process",
  "DMC",  "$\\tau|DMC$",    "$\\tau$",    "Inhibitory process",
  "DSTP", "$a|DSTP$",       "$a$",        "Decision Caution",
  "DSTP", "$t|DSTP$",       "$t$",        "Non-decision time",
  "DSTP", "$v_{ta}|DSTP$",  "$v_{ta}$",   "Processing Efficiency",
  "DSTP", "$v_{fl}|DSTP$",  "$v_{fl}$",   "Inhibitory process",
  "DSTP", "$v_{ss}|DSTP$",  "$v_{ss}$",   "Processing Efficiency",
  "DSTP", "$v_{p2}|DSTP$",  "$v_{p2}$",   "Decision Caution",
  "DSTP", "$a_{ss}|DSTP$",  "$a_{ss}$",   "Processing Efficiency",
  "SSP",  "$a|SSP$",        "$a$",        "Decision Caution",
  "SSP",  "$t|SSP$",        "$t$",        "Non-decision time",
  "SSP",  "$p|SSP$",        "$p$",        "Processing Efficiency",
  "SSP",  "$r_d|SSP$",      "$r_d$",      "Processing Efficiency",
  "SSP",  "$sd_a|SSP$",     "$sd_a$",     "Inhibitory process",
  "DDM",  "$a|DDM$",        "$a$",        "Decision Caution",
  "DDM",  "$t|DDM$",        "$t$",        "Non-decision time",
  "DDM",  "$v_{cong}|DDM$", "$v_{cong}$", "Processing Efficiency",
  "DDM",  "$v_{incong}|DDM$","$v_{incong}$","Processing Efficiency"
)

CONSISTENCY_COLORS <- c(
  "Matched"      = "#4CAF50",
  "Mismatched"   = "#8491B4",
  "Inconsistent" = "#9E9E9E"
)

RETEST_LABS <- c("hedge2018", "eisenberg2019", "clayson2024", "lee2025")

# --- helpers -----------------------------------------------------------------
add_param_metadata <- function(df) {
  join_cols <- if ("model" %in% names(df)) c("param" = "param_raw", "model") else c("param" = "param_raw")
  df %>% left_join(PARAM_CONFIG, by = join_cols)
}

# Reconstructed: two-occasion (s1/s2) test-retest ICC(2,1).
compute_retest_icc <- function(df_sub) {
  RETEST_SESSIONS <- c("s1", "s2")
  df_pair <- df_sub %>% filter(session_id %in% RETEST_SESSIONS)
  n_sess <- n_distinct(df_pair$session_id)
  n_subj <- n_distinct(df_pair$subject_id)
  if (n_sess < 2 || n_subj < 5) {
    return(tibble(overall_icc = NA_real_, ci_lower = NA_real_, ci_upper = NA_real_))
  }
  mat <- df_pair %>%
    select(subject_id, session_id, value) %>%
    pivot_wider(names_from = session_id, values_from = value) %>%
    select(-subject_id)
  mat <- mat[complete.cases(mat), , drop = FALSE]
  if (nrow(mat) < 5) {
    return(tibble(overall_icc = NA_real_, ci_lower = NA_real_, ci_upper = NA_real_))
  }
  res <- tryCatch(psych::ICC(mat, lmer = FALSE), error = function(e) NULL)
  if (is.null(res)) return(tibble(overall_icc = NA_real_, ci_lower = NA_real_, ci_upper = NA_real_))
  icc_row <- res$results[res$results$type == "ICC2", ]
  tibble(overall_icc = icc_row$ICC, ci_lower = icc_row$`lower bound`, ci_upper = icc_row$`upper bound`)
}

run_retest_icc_analysis <- function(df_data, pool_authors = FALSE) {
  df_processed <- df_data %>%
    mutate(global_subject_id = paste(author_year, subject_id, sep = "_")) %>%
    mutate(subject_id = global_subject_id)
  group_vars <- if (pool_authors) {
    c("task_name", "model", "param", "consistency_group")
  } else {
    c("author_year", "task_name", "model", "param", "consistency_group")
  }
  df_processed %>%
    group_by(across(all_of(group_vars))) %>%
    group_modify(~ compute_retest_icc(.x)) %>%
    ungroup() %>%
    mutate(consistency_group = factor(consistency_group, levels = c("Matched", "Mismatched", "Inconsistent"))) %>%
    mutate(across(where(is.numeric), ~ round(.x, digits = 3)))
}

# --- data-prep-retest (Rmd lines 422-474) ------------------------------------
df_raw_retest  <- read.csv(file.path(INTERMEDIATE_DIR, "factor_scores_retest.csv"), check.names = FALSE)
df_rmse_retest <- read.csv(file.path(INTERMEDIATE_DIR, "model_prediction_indices_retest.csv"), check.names = FALSE) %>%
  mutate(across(c(author_year, task_name), tolower))

# Winning model per session -> RAP (Relative Agreement Proportion)
df_win_counts <- df_rmse_retest %>%
  filter(author_year %in% RETEST_LABS) %>%
  group_by(subject_id, author_year, task_name, session_id) %>%
  slice_min(order_by = RMSE, n = 1, with_ties = FALSE) %>%
  group_by(subject_id, author_year, task_name) %>%
  mutate(total_sessions = n_distinct(session_id)) %>%
  group_by(subject_id, author_year, task_name, model, total_sessions) %>%
  summarise(win_count = n(), .groups = "drop") %>%
  mutate(RAP = win_count / total_sessions)

# Consistency group per subject x task (consistent if a model wins >= 70% sessions)
df_consistency_retest <- df_win_counts %>%
  group_by(subject_id, author_year, task_name) %>%
  slice_max(order_by = RAP, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  mutate(
    is_consistent    = RAP >= 0.7,
    consistent_model = if_else(is_consistent, model, "Inconsistent")
  ) %>%
  select(subject_id, author_year, task_name, consistent_model, max_RAP = RAP)

df_analysis_retest <- df_raw_retest %>%
  filter(author_year %in% RETEST_LABS) %>%
  select(subject_id, author_year, task_name, session_id, any_of(PARAM_CONFIG$param_raw)) %>%
  pivot_longer(cols = any_of(PARAM_CONFIG$param_raw), names_to = "param", values_to = "value") %>%
  filter(!is.na(value)) %>%
  add_param_metadata() %>%
  inner_join(df_consistency_retest, by = c("subject_id", "author_year", "task_name")) %>%
  mutate(consistency_group = case_when(
    consistent_model == model          ~ "Matched",
    consistent_model == "Inconsistent" ~ "Inconsistent",
    TRUE                               ~ "Mismatched"
  )) %>%
  # S15 uses only DMC + DSTP panels; restricting here is equivalent and faster.
  filter(model %in% c("DMC", "DSTP"))

# --- calc-stratified-icc (Rmd lines 476-523) ---------------------------------
icc_retest_pooled <- run_retest_icc_analysis(df_analysis_retest, pool_authors = TRUE)

# --- plot-retest-consistency (Rmd lines 525-605) -----------------------------
plot_icc_retest <- function(df_icc, target_task) {
  df_plot <- df_icc %>%
    filter(task_name == target_task) %>%
    mutate(
      consistency_group = factor(consistency_group, levels = c("Matched", "Mismatched", "Inconsistent")),
      param_clean = str_extract(param, "^[^|]+")
    ) %>%
    filter(!is.na(overall_icc))

  ggplot(df_plot, aes(x = overall_icc, y = param_clean, color = consistency_group)) +
    geom_vline(xintercept = 0, linetype = "solid", color = "gray80") +
    geom_vline(xintercept = c(0.5, 0.75), linetype = "dashed", color = "gray70") +
    geom_pointrange(aes(xmin = ci_lower, xmax = ci_upper),
                    position = position_dodge(width = 0.6), size = 0.7, linewidth = 1) +
    facet_wrap(~ model, scales = "free_y", ncol = 1) +
    scale_color_manual(values = CONSISTENCY_COLORS) +
    scale_y_discrete(labels = function(x) unname(TeX(x))) +
    scale_x_continuous(limits = c(-0.1, 1), breaks = seq(0, 1, 0.25)) +
    labs(
      title = str_to_title(target_task),
      x = expression("ICC"["(2,1)"] ~ " [95% CI]"),
      y = "",
      color = "Consistency Group"
    ) +
    theme_bw(base_size = 14) +
    theme(
      strip.background = element_rect(fill = "#f0f0f0", color = "black"),
      strip.text = element_text(face = "bold", size = 12),
      panel.grid.minor = element_blank(),
      plot.title = element_text(hjust = 0.5, face = "bold")
    )
}

p_task1 <- plot_icc_retest(icc_retest_pooled, "flanker")
p_task2 <- plot_icc_retest(icc_retest_pooled, "simon")
p_task3 <- plot_icc_retest(icc_retest_pooled, "stroop")

p_combined_tasks <- (p_task1 | p_task2 | p_task3) + plot_layout(guides = "collect")

# Save at the chunk's display geometry (14 x 10 -> AR 1.4, matches image15.png).
ggsave(OUT_SVG, p_combined_tasks, width = 14, height = 10, device = "svg")
ggsave(OUT_PNG, p_combined_tasks, width = 14, height = 10, dpi = 300)
message("Saved: ", OUT_SVG)
message("Saved: ", OUT_PNG)
