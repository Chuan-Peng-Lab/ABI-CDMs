# ==============================================================================
# Script: 44_viz_functions.R
# Purpose: Visualization functions for factor analysis report (44_factor_analysis.Rmd).
#          Source this file after 44_fitting_models.R.
# ==============================================================================

# Canonical factor display names (used by plot_icc_ridge and plot_icc_forest)
FACTOR_DISPLAY <- c(
  Control           = "Processing Efficiency",
  Decision_Caution  = "Decision Caution",
  Inhibitory        = "Inhibitory process",
  Non_decision_time = "Non-decision time"
)

# Default factor order for plots (by theoretical grouping)
FACTOR_ORDER <- c("Non-decision time", "Processing Efficiency",
                  "Decision Caution",  "Inhibitory process")


# 1. plot_petal_grid -----------------------------------------------------------

#' Generate Petal Plots (Polar Faceted Bar Charts)
#'
#' @param summary_data Aggregated dataframe (must have mean_val, ci_lower, ci_upper).
#' @param x_var Variable for petal categories (default: "Factor_Type").
#' @param y_var Variable for petal length (default: "mean_val").
#' @param row_facet Row facet variable (default: "author_year").
#' @param col_facet Column facet variable (default: "task_name").
#' @param fill_var Fill color variable (default: "Factor_Type").
#' @param colors Named color vector; auto-generated if NULL.
#' @param title Optional plot title.
#' @param save_name Optional SVG filepath.
plot_petal_grid <- function(summary_data,
                            x_var     = "Factor_Type",
                            y_var     = "mean_val",
                            row_facet = "author_year",
                            col_facet = "task_name",
                            fill_var  = "Factor_Type",
                            colors    = NULL,
                            title     = NULL,
                            save_name = NULL) {
  global_min <- min(summary_data$ci_lower, na.rm = TRUE) - 0.1
  global_max <- max(summary_data$ci_upper, na.rm = TRUE)

  plot_data <- summary_data %>%
    mutate(
      x_num = as.numeric(as.factor(!!sym(x_var))),
      xmin  = x_num - 0.45,
      xmax  = x_num + 0.45
    )

  if (is.null(colors)) {
    lvl    <- unique(summary_data[[fill_var]])
    colors <- setNames(scales::hue_pal()(length(lvl)), lvl)
  }

  p <- ggplot(plot_data) +
    geom_rect(aes(xmin = xmin, xmax = xmax,
                  ymin = global_min, ymax = !!sym(y_var),
                  fill = !!sym(fill_var)),
              color = "black", linewidth = 0.15) +
    geom_errorbar(aes(x = x_num, ymin = ci_lower, ymax = ci_upper),
                  width = 0.2, linewidth = 0.3) +
    coord_polar(theta = "x", start = 0, clip = "off") +
    facet_grid(rows = vars(!!sym(row_facet)),
               cols = vars(!!sym(col_facet)), switch = "y") +
    scale_y_continuous(limits = c(global_min, global_max)) +
    scale_x_continuous(limits = c(0.5, max(plot_data$x_num) + 0.5)) +
    scale_fill_manual(values = colors) +
    theme_minimal(base_size = 14) +
    labs(title = title) +
    theme(
      axis.title = element_blank(), axis.text = element_blank(),
      axis.ticks = element_blank(), panel.grid = element_blank(),
      panel.spacing = unit(0.5, "lines"),
      strip.background = element_blank(),
      strip.text.x = element_text(face = "bold", size = 11, margin = margin(b = 5)),
      strip.text.y.left = element_text(face = "bold", size = 10, angle = 0, hjust = 1),
      legend.position = "bottom", legend.title = element_blank(),
      legend.key.size = unit(0.4, "cm"), legend.text = element_text(size = 9),
      legend.margin = margin(t = 0),
      plot.title = element_text(hjust = 0.5, face = "bold"),
      plot.margin = margin(10, 10, 10, 10)
    )

  if (!is.null(save_name)) {
    n_rows     <- length(unique(summary_data[[row_facet]]))
    calc_height <- max(4, n_rows * 1.2)
    ggsave(save_name, p, width = 8, height = calc_height, limitsize = FALSE)
    message("Saved: ", save_name)
  }
  p
}


# 2. plot_factor_comparison ----------------------------------------------------

#' Plot Grouped Factors with Individual Points and Paired Lines
#'
#' @param data Cleaned summary dataframe.
#' @param x_var Categorical x-axis variable.
#' @param y_var Numeric y-axis variable.
#' @param group_var Grouping/color variable (e.g., "task").
#' @param point_label_var Variable for pairing lines (e.g., "author_year").
#' @param colors Named color vector; auto-generated if NULL.
#' @param x_order Factor level order for x-axis.
#' @param x_tick_angle X-axis label rotation angle.
#' @param save_name Optional SVG filepath.
#' @param dodge_width Dodging width for grouped geoms.
plot_factor_comparison <- function(data,
                                   x_var           = "Factor_Type",
                                   y_var           = "mean_val",
                                   group_var       = "task",
                                   point_label_var = "author_year",
                                   colors          = NULL,
                                   x_order         = FACTOR_ORDER,
                                   x_tick_angle    = 0,
                                   save_name       = NULL,
                                   dodge_width     = 0.8) {
  plot_df <- data %>%
    filter(!!sym(x_var) %in% x_order) %>%
    mutate(!!sym(x_var) := factor(!!sym(x_var), levels = x_order))

  groups   <- sort(unique(plot_df[[group_var]]))
  n_groups <- length(groups)
  offset_map <- tibble(
    !!sym(group_var) := groups,
    offset = (seq_along(groups) - (n_groups + 1) / 2) * (dodge_width / n_groups)
  )

  plot_df <- plot_df %>%
    left_join(offset_map, by = group_var) %>%
    mutate(x_numeric = as.numeric(!!sym(x_var)),
           x_final   = x_numeric + offset)

  if (is.null(colors))
    colors <- setNames(scales::hue_pal()(length(groups)), groups)

  p <- ggplot(plot_df, aes(x = !!sym(x_var), y = !!sym(y_var),
                           fill = !!sym(group_var))) +
    geom_boxplot(aes(color = !!sym(group_var)),
                 position = position_dodge(width = dodge_width),
                 width = 0.6, alpha = 0.1, outlier.shape = NA) +
    geom_line(aes(x = x_final, y = !!sym(y_var),
                  group = interaction(!!sym(point_label_var), !!sym(x_var))),
              color = "grey20", alpha = 0.3, linewidth = 0.4) +
    geom_point(aes(x = x_final, color = !!sym(group_var)),
               size = 2, alpha = 0.9) +
    scale_color_manual(values = colors) +
    scale_fill_manual(values  = colors) +
    theme_classic(base_size = 14) +
    labs(y = "Standardized Value") +
    theme(
      legend.position = "right", legend.title = element_blank(),
      legend.text = element_text(face = "bold", size = 11),
      axis.title.x = element_blank(),
      axis.text.y  = element_text(color = "black"),
      axis.line    = element_line(linewidth = 0.5),
      axis.text.x  = element_text(face = "bold", color = "black", size = 11,
                                  angle = x_tick_angle,
                                  hjust = if (x_tick_angle > 0) 1 else 0.5,
                                  vjust = if (x_tick_angle > 0) 1 else 0.5,
                                  margin = margin(t = 10)),
      panel.grid.major.y = element_line(color = "grey92", linetype = "dashed")
    )

  if (!is.null(save_name)) {
    ggsave(save_name, p, width = 7, height = 5, device = "svg")
    message("Saved: ", save_name)
  }
  p
}


# 3. plot_icc_ridge ------------------------------------------------------------

#' Plot ICC Posterior Distributions as Smooth Ridgelines
#'
#' Expects Factor_Type labels in canonical short form (Control, Decision_Caution,
#' Inhibitory, Non_decision_time) as produced by get_posterior_icc().
#' Display names are applied via FACTOR_DISPLAY internally.
#'
#' @param icc_data Tibble from get_posterior_icc() with an ICC column.
#' @param fill_colors Named color vector (names = display factor names).
#' @param icc_col Name of the ICC column (default: "ICC").
#' @param x_title X-axis label.
#' @param x_limits Numeric length-2 vector for x-axis limits.
#' @param save_name Optional SVG filepath.
plot_icc_ridge <- function(icc_data,
                           fill_colors,
                           icc_col   = "ICC",
                           x_title   = "Intra-Class Correlation (ICC)",
                           x_limits  = c(-0.03, 0.3),
                           save_name = NULL) {
  # Apply canonical -> display name mapping
  icc_data <- icc_data %>%
    mutate(Factor_Type = recode(Factor_Type, !!!FACTOR_DISPLAY))

  # Reorder by median ICC (ascending, so highest is on top in ridgeline)
  factor_order <- icc_data %>%
    group_by(Factor_Type) %>%
    summarise(med = median(.data[[icc_col]]), .groups = "drop") %>%
    arrange(med) %>%
    pull(Factor_Type)

  p <- icc_data %>%
    mutate(Factor_Type = factor(Factor_Type, levels = factor_order)) %>%
    ggplot(aes(x = .data[[icc_col]], y = Factor_Type, fill = Factor_Type)) +
    geom_density_ridges(scale = 1.2, alpha = 0.7, color = "white",
                        rel_min_height = 0.005) +
    stat_pointinterval(aes(y = as.numeric(Factor_Type)),
                       .width = 0.95, point_color = "black",
                       interval_color = "black",
                       position = position_nudge(y = -0.15)) +
    scale_fill_manual(values = fill_colors) +
    scale_x_continuous(limits = x_limits,
                       breaks = seq(0, x_limits[2], 0.1),
                       expand = c(0, 0)) +
    labs(x = x_title, y = NULL) +
    theme_minimal(base_size = 14) +
    theme(
      legend.position = "none",
      axis.text.y     = element_text(size = 11, face = "bold", color = "black"),
      axis.title.x    = element_text(size = 12, margin = margin(t = -10)),
      axis.text.x     = element_text(size = 12),
      panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank(),
      panel.grid.major.y = element_line(color = "#ebebeb")
    )

  if (!is.null(save_name)) {
    ggsave(save_name, plot = p, width = 6, height = 4)
    message("Saved: ", save_name)
  }
  p
}


# 4. plot_diff_heatmap ---------------------------------------------------------

#' Plot Pairwise ICC Difference Heatmap (Lower Triangle)
#'
#' @param icc_data Tibble from get_posterior_icc() with Factor_Type and ICC column.
#' @param icc_col Name of ICC column (default: "ICC").
#' @param save_name Optional SVG filepath.
plot_diff_heatmap <- function(icc_data, icc_col = "ICC", save_name = NULL) {
  factors <- unique(icc_data$Factor_Type)
  factor_levels <- sort(factors)

  diff_df <- map_df(factors, function(f1) {
    map_df(factors, function(f2) {
      if (f1 == f2) return(NULL)
      vec1 <- icc_data %>% filter(Factor_Type == f1) %>% pull(!!sym(icc_col))
      vec2 <- icc_data %>% filter(Factor_Type == f2) %>% pull(!!sym(icc_col))
      dv   <- vec1 - vec2
      tibble(Factor1 = f1, Factor2 = f2,
             mean_diff = mean(dv), pd = mean(dv > 0),
             sig = !(quantile(dv, 0.025) < 0 & quantile(dv, 0.975) > 0))
    })
  })

  plot_df <- diff_df %>%
    mutate(f1_idx = match(Factor1, factor_levels),
           f2_idx = match(Factor2, factor_levels)) %>%
    filter(f1_idx > f2_idx) %>%
    mutate(label_text = paste0(sprintf("%.2f", mean_diff), "\n",
                               "(", scales::percent(pd, accuracy = 0.1), ")"))

  p <- ggplot(plot_df, aes(x = Factor2, y = Factor1, fill = mean_diff)) +
    geom_tile(color = "white") +
    geom_text(aes(label = label_text), size = 4, color = "black") +
    scale_fill_gradient2(low = "#4575b4", mid = "#f7f7f7", high = "#d73027",
                         midpoint = 0, name = expression(Delta ~ ICC)) +
    scale_x_discrete(position = "bottom") +
    labs(title    = "Pairwise Comparison of ICC Differences",
         subtitle = "Values: Mean Difference (Probability A > B)",
         x = NULL, y = NULL) +
    theme(panel.grid = element_blank(),
          axis.text.x = element_text(angle = 45, hjust = 1))

  if (!is.null(save_name)) {
    ggsave(save_name, plot = p, device = "svg", width = 7, height = 6)
    message("Saved: ", save_name)
  }
  p
}


# 5. plot_icc_forest -----------------------------------------------------------

#' Plot Forest of Test-Retest ICCs Across Tasks and Studies
#'
#' @param data Subgroup ICC dataframe from calculate_subgroup_iccs().
#' @param color_palette Named color vector for Factor_Type.
#' @param x_order Factor level order for faceting.
#' @param save_name Optional SVG filepath.
plot_icc_forest <- function(data,
                            color_palette = NULL,
                            x_order       = FACTOR_ORDER,
                            save_name     = NULL) {
  data <- data %>%
    mutate(task_name = tools::toTitleCase(as.character(task_name)))

  # Fix: assign result of mutate (was a no-op in original)
  existing_levels  <- intersect(x_order, unique(data$Factor_Type))
  data$Factor_Type <- factor(data$Factor_Type, levels = existing_levels)

  group_means <- data %>%
    dplyr::group_by(Factor_Type) %>%
    dplyr::summarise(mean_icc = mean(ICC, na.rm = TRUE), .groups = "drop")

  p <- ggplot(data, aes(x = ICC, y = task_name, color = Factor_Type)) +
    geom_vline(data = group_means,
               aes(xintercept = mean_icc, color = Factor_Type),
               linewidth = 1.2, alpha = 0.8) +
    geom_jitter(height = 0.2, width = 0, size = 3, alpha = 0.6) +
    facet_wrap(~Factor_Type, ncol = 1, scales = "free_y") +
    scale_color_manual(values = color_palette) +
    scale_x_continuous(limits = c(0, 0.8), breaks = seq(0, 0.8, 0.2)) +
    labs(x = "Cross-temporal ICC", y = NULL, color = NULL) +
    theme_minimal(base_size = 14) +
    theme(
      legend.position = "none",
      panel.grid.minor = element_blank(),
      strip.text = element_text(face = "bold", size = 12, hjust = 0),
      axis.text.y = element_text(face = "bold")
    )

  if (!is.null(save_name)) {
    ggsave(save_name, p, width = 8, height = 10, device = "svg")
    message("Saved: ", save_name)
  }
  p
}


# 6. plot_sd_comparison --------------------------------------------------------

#' Plot Posterior SD Distributions for Tasks vs. Labs
#'
#' @param model Fitted brms meta-regression model (fit_reliability_meta_model).
#' @param x_lims X-axis range.
#' @param colors Named color vector for "Tasks" and "Labs".
#' @param labels Factor level order for y-axis.
plot_sd_comparison <- function(model,
                               x_lims = c(0, 0.8),
                               colors = c("Tasks" = "#95d8c3", "Labs" = "#81cef0"),
                               labels = c("Labs", "Tasks")) {
  post_sd <- model %>%
    spread_draws(sd_author_year__Intercept, sd_task_name__Intercept) %>%
    rename(Labs = sd_author_year__Intercept, Tasks = sd_task_name__Intercept) %>%
    pivot_longer(c(Labs, Tasks), names_to = "Source", values_to = "Sigma") %>%
    mutate(Source = factor(Source, levels = labels))

  ggplot(post_sd, aes(x = Sigma, y = Source, fill = Source)) +
    stat_halfeye(alpha = 0.8) +
    scale_fill_manual(values = colors) +
    scale_x_continuous(breaks = seq(0, 0.8, 0.2)) +
    coord_cartesian(xlim = x_lims) +
    labs(x = "Estimated SD", y = NULL) +
    theme_minimal() +
    theme(
      legend.position = "none",
      axis.text.y = element_text(face = "bold", size = 12),
      axis.title.x = element_text(size = 14),
      panel.grid.major.y = element_blank(), panel.grid.minor.y = element_blank(),
      panel.grid.minor.x = element_blank()
    )
}


# 7. plot_sd_difference --------------------------------------------------------

#' Plot Posterior Distribution of SD Difference (Lab - Task)
#'
#' @param model Fitted brms meta-regression model.
#' @param x_lims X-axis range.
#' @param fill_color Fill color for the slab.
plot_sd_difference <- function(model,
                               x_lims     = c(-1, 1),
                               fill_color = "grey85") {
  post_diff <- model %>%
    spread_draws(sd_author_year__Intercept, sd_task_name__Intercept) %>%
    mutate(diff = sd_author_year__Intercept - sd_task_name__Intercept)

  prob_greater <- mean(post_diff$diff > 0)

  ggplot(post_diff, aes(x = diff)) +
    stat_halfeye(fill = fill_color, alpha = 0.7) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "black",
               linewidth = 0.8) +
    annotate("text", x = 0.05, y = 0.9,
             label = paste0("P(Lab > Task) = ", round(prob_greater * 100, 1), "%"),
             hjust = 0, fontface = "bold") +
    scale_x_continuous(breaks = seq(-1, 1, 0.5)) +
    coord_cartesian(xlim = x_lims) +
    labs(x = "Difference in SD (Lab - Task)", y = "") +
    theme_minimal() +
    theme(
      axis.text.y = element_blank(), axis.ticks.y = element_blank(),
      panel.grid.major.y = element_blank(), panel.grid.minor.y = element_blank(),
      panel.grid.minor.x = element_blank()
    )
}
