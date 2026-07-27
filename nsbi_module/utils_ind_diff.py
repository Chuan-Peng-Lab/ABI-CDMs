import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import entropy
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from typing import Dict, Any, Optional
try:
    import pingouin as pg
except ImportError:
    pg = None
import matplotlib.patches as mpatches
import re

from .study_labels import format_author_year


def format_task_name(text):
    return text.capitalize()

def format_task_id(text):
    parts = re.findall(r'[a-zA-Z]+|\d+', text)
    formatted_parts = [part.capitalize() if part.isalpha() else part for part in parts]
    return ' '.join(formatted_parts)

def get_best_model_by_metric(
    df_long: pd.DataFrame,
    metric: str = "KL",
    drop_nan: bool = True,
    group=["subject_id", "task_id", "author_year", "task_name"],
    model_col="model",
) -> pd.DataFrame:
    """
    For each subject, find the model with the lowest value for the given metric.

    Parameters
    ----------
    df_long : pd.DataFrame
        Long-format DataFrame with columns including 'subject_id', 'model', and the metric.
    metric : str
        The metric to use for model selection (e.g., 'aBIC', 'KL').
    drop_nan : bool
        Whether to drop rows with NaN in the metric column.

    Returns
    -------
    pd.DataFrame
        DataFrame with the best model per subject (lowest metric value).
    """
    # Remove rows with 0 or NaN in the metric column
    df_metric = df_long.copy()
    if drop_nan:
        df_metric = df_metric.dropna(subset=[metric])
    idx = df_metric.groupby(group)[metric].idxmin()
    best_models = df_metric.loc[idx, group + [model_col]].reset_index(drop=True)
    # Rename model_col column to the metric name
    best_models = best_models.rename(columns={model_col: "winner_model"})
    return best_models
def calc_best_model_proportion(
    df: pd.DataFrame, group_cols=["task_id", "task_name"], model_col="winner_model"
) -> pd.DataFrame:
    """
    Calculate the proportion of each best model within each group.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least the group columns and the model column.
    group_cols : list of str
        Columns to group by (e.g., task_id, task_name).
    model_col : str
        Column name for the best model label.

    Returns
    -------
    pd.DataFrame
        DataFrame with group columns, model names, counts, and their proportions within each group.
        Columns: group_cols + [model_col, 'count', 'proportion', 'n_subj']
    """
    count_df = df.groupby(group_cols + [model_col]).size().reset_index(name="count")
    total_df = count_df.groupby(group_cols)["count"].transform("sum")
    count_df["proportion"] = count_df["count"] / total_df
    count_df["n_subj"] = total_df
    return count_df

def apply_nature_bar_axis_style(
    ax,
    xlabel: str = "",
    ylabel: str = "",
    label_fontsize: int = 17,
    tick_fontsize: int = 14,
    spine_width: float = 1.6,
    grid: bool = False,
    grid_axis: str = "y",
):
    """
    Apply a restrained publication-style axis treatment for bar plots.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to style.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    label_fontsize : int
        Axis label font size.
    tick_fontsize : int
        Tick label font size.
    spine_width : float
        Width for the visible left and bottom spines.
    grid : bool
        Whether to show a light reference grid.
    grid_axis : str
        Axis for the reference grid.

    Returns
    -------
    matplotlib.axes.Axes
        Styled axis.
    """
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize, width=spine_width, length=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(spine_width)
    ax.spines["bottom"].set_linewidth(spine_width)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(True, axis=grid_axis, color="#D9D9D9", linewidth=0.8, alpha=0.7)
    else:
        ax.grid(False)
    return ax


def order_groups_by_dominant_model(
    proportion_df: pd.DataFrame,
    group_col: str,
    model_col: str = "winner_model",
    proportion_col: str = "proportion",
    dominant_model_priority=("DMC", "DSTP"),
):
    """
    Order groups by dominant model family and descending dominant proportion.

    Parameters
    ----------
    proportion_df : pd.DataFrame
        Long-format proportions with one row per group and model.
    group_col : str
        Column containing the group labels to order.
    model_col : str
        Column containing model names.
    proportion_col : str
        Column containing model proportions.
    dominant_model_priority : tuple
        Dominant models that should appear first, in order.

    Returns
    -------
    list
        Ordered group labels.
    """
    pivot_df = proportion_df.pivot(
        index=group_col, columns=model_col, values=proportion_col
    ).fillna(0)
    order_df = pd.DataFrame(
        {
            group_col: pivot_df.index,
            "dominant_model": pivot_df.idxmax(axis=1).to_numpy(),
            "dominant_proportion": pivot_df.max(axis=1).to_numpy(),
        }
    )
    priority = {model: i for i, model in enumerate(dominant_model_priority)}
    fallback_priority = len(priority)
    order_df["dominant_priority"] = order_df["dominant_model"].map(priority).fillna(
        fallback_priority
    )
    order_df = order_df.sort_values(
        ["dominant_priority", "dominant_model", "dominant_proportion", group_col],
        ascending=[True, True, False, True],
    )
    return order_df[group_col].tolist()


def order_groups_by_model_proportions(
    proportion_df: pd.DataFrame,
    group_col: str,
    sort_models=("DSTP", "DMC"),
    model_col: str = "winner_model",
    proportion_col: str = "proportion",
):
    """
    Order groups by the descending proportions of specified models.

    Parameters
    ----------
    proportion_df : pd.DataFrame
        Long-format proportions with one row per group and model.
    group_col : str
        Column containing the group labels to order.
    sort_models : tuple
        Model columns used as descending sort keys, in priority order.
    model_col : str
        Column containing model names.
    proportion_col : str
        Column containing model proportions.

    Returns
    -------
    list
        Ordered group labels.
    """
    pivot_df = proportion_df.pivot(
        index=group_col, columns=model_col, values=proportion_col
    ).fillna(0)
    for model in sort_models:
        if model not in pivot_df.columns:
            pivot_df[model] = 0
    sort_columns = list(sort_models) + [group_col]
    order_df = pivot_df.reset_index().sort_values(
        sort_columns,
        ascending=[False] * len(sort_models) + [True],
    )
    return order_df[group_col].tolist()


def plot_stacked_model_proportion_bar(
    proportion_df: pd.DataFrame,
    colors,
    group_col: str,
    model_col: str = "winner_model",
    proportion_col: str = "proportion",
    models_sorted=("DSTP", "DMC", "SSP", "DDM"),
    group_order=None,
    edge_color=None,
    alpha: float = 1,
    width: float = 0.72,
    ax=None,
    xlabel: str = "",
    ylabel: str = "Percentage of subjects (%)",
    rotate_x_labels=0,
    label_fontsize: int = 17,
    tick_fontsize: int = 14,
    show_legend: bool = False,
    show_segment_labels: bool = True,
    label_threshold: float = 15,
    segment_label_fontsize: int = 9,
    show_n_prefix: bool = True,
):
    """
    Plot precomputed best-model proportions as vertical stacked percentage bars.

    Parameters
    ----------
    proportion_df : pd.DataFrame
        Long-format proportions with one row per group and model.
    colors : dict
        Mapping from model name to bar fill color.
    group_col : str
        Column used for x-axis groups.
    model_col : str
        Column containing model names.
    proportion_col : str
        Column containing model proportions in 0-1 units.
    models_sorted : tuple
        Stack and legend order for models.
    group_order : list or None
        Explicit x-axis group order. If None, groups are sorted alphabetically.
    edge_color : dict or str or None
        Optional edge color mapping or a single edge color.
    alpha : float
        Bar opacity.
    width : float
        Bar width.
    ax : matplotlib.axes.Axes or None
        Axis to plot on. If None, creates a new figure and axis.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    rotate_x_labels : int or float
        X tick label rotation in degrees.
    label_fontsize : int
        Axis label font size.
    tick_fontsize : int
        Tick label font size.
    show_legend : bool
        Whether to draw an axis-level legend.
    show_segment_labels : bool
        Whether to draw percentage and count labels within large bar segments.
    label_threshold : float
        Minimum percentage for drawing a segment label. Labels are drawn only above this value.
    segment_label_fontsize : int
        Font size for labels inside bar segments.

    Returns
    -------
    matplotlib.axes.Axes
        Axis with the stacked percentage bars.
    """
    if group_order is None:
        group_order = sorted(proportion_df[group_col].unique())

    plot_df = proportion_df.copy()
    if "count" not in plot_df.columns and "n_subj" in plot_df.columns:
        plot_df["count"] = (plot_df[proportion_col] * plot_df["n_subj"]).round().astype(int)

    pivot_df = plot_df.pivot(
        index=group_col, columns=model_col, values=proportion_col
    ).fillna(0)
    pivot_df = pivot_df.reindex(index=group_order, columns=models_sorted, fill_value=0)
    percentage_df = pivot_df * 100
    count_df = None
    if "count" in plot_df.columns:
        count_df = plot_df.pivot(index=group_col, columns=model_col, values="count").fillna(0)
        count_df = count_df.reindex(index=group_order, columns=models_sorted, fill_value=0)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(percentage_df.index))
    bottom = np.zeros(len(percentage_df.index))
    for model in models_sorted:
        values = percentage_df[model].to_numpy()
        if isinstance(edge_color, dict):
            ec = edge_color.get(model, "white")
        elif edge_color is None:
            ec = "white"
        else:
            ec = edge_color
        ax.bar(
            x,
            values,
            width=width,
            bottom=bottom,
            label=model,
            color=colors.get(model, "#BDBDBD"),
            edgecolor=ec,
            linewidth=1.0,
            alpha=alpha,
        )
        if show_segment_labels:
            counts = (
                count_df[model].to_numpy()
                if count_df is not None
                else np.full(len(values), np.nan)
            )
            for x_pos, value, count, base in zip(x, values, counts, bottom):
                if value > label_threshold:
                    if np.isnan(count):
                        label = f"{value:.1f}%"
                    else:
                        if show_n_prefix:
                            label = f"{value:.1f}%\n(n={int(count)})"
                        else:
                            label = f"{value:.1f}%\n({int(count)})"
                    ax.text(
                        x_pos,
                        base + value / 2,
                        label,
                        ha="center",
                        va="center",
                        fontsize=segment_label_fontsize,
                        color="black",
                        linespacing=0.9,
                    )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(percentage_df.index)
    if rotate_x_labels:
        ax.tick_params(axis="x", rotation=rotate_x_labels)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 25))
    apply_nature_bar_axis_style(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
    )
    if show_legend:
        ax.legend(title="", frameon=False, fontsize=tick_fontsize)
    return ax


def plot_best_model_proportion_barh(
    proportion_df: pd.DataFrame,
    colors,
    group_col: str = "task_id",
    model_col: str = "winner_model",
    proportion_col: str = "proportion",
    models_sorted=["DDM","SSP","DMC","DSTP"],
    order_by: str = None,
    group_order=None,
    alpha=1,
    figsize=(8, 6),
    ax=None,
    show_percentage: bool = True,  # New parameter to toggle percentage display
    label_threshold: float = 0.15,
):
    """
    Plot a horizontal bar chart showing the proportion of best models for each group.

    Parameters
    ----------
    proportion_df : pd.DataFrame
        DataFrame with columns including group_col, model_col, and proportion_col.
    group_col : str
        The column to use as the y-axis (default: "task_id").
    model_col : str
        The column indicating the model name (default: "winner_model").
    proportion_col : str
        The column indicating the proportion (default: "proportion").
    order_by : str or None
        Optional column to sort the y-axis (e.g., "task_name", "author_year"). If None, sort by group_col.
    group_order : list or None
        Explicit y-axis group order. Takes precedence over order_by.
    figsize : tuple
        Figure size.
    palette : str or list
        Color palette for models.
    ax : matplotlib.axes.Axes or None
        Axis to plot on. If None, creates a new figure.
    show_percentage : bool
        Whether to display percentage labels on the bars (default: True).
    label_threshold : float
        Minimum proportion for drawing a segment label. Labels are drawn only above this value.
    Returns
    -------
    matplotlib.axes.Axes
        The axis with the plot.
    """
    import matplotlib.pyplot as plt

    # Determine order
    if group_order is not None:
        order = group_order
    elif order_by and order_by in proportion_df.columns:
        order = (
            proportion_df.drop_duplicates([group_col, order_by])
            .sort_values(order_by)[group_col]
            .tolist()
        )
    else:
        order = sorted(proportion_df[group_col].unique())

    # Pivot for stacked barh
    plot_df = proportion_df.copy()
    if "count" not in plot_df.columns and "n_subj" in plot_df.columns:
        plot_df["count"] = (plot_df[proportion_col] * plot_df["n_subj"]).round().astype(int)

    pivot_df = plot_df.pivot(
        index=group_col, columns=model_col, values=proportion_col
    ).fillna(0)
    pivot_df = pivot_df.reindex(index=order, columns=models_sorted, fill_value=0)
    count_df = None
    if "count" in plot_df.columns:
        count_df = plot_df.pivot(index=group_col, columns=model_col, values="count").fillna(0)
        count_df = count_df.reindex(index=order, columns=models_sorted, fill_value=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    left = np.zeros(len(pivot_df))

    for model in models_sorted:
        bars = ax.barh(
            pivot_df.index,
            pivot_df[model],
            left=left,
            label=model,
            color=colors.get(model, "#BDBDBD"),
            edgecolor="white",
            linewidth=1.0,
            alpha=alpha,
        )
        left += pivot_df[model].to_numpy()

        # Add percentage annotations
        if show_percentage:
            for bar, idx in zip(bars, pivot_df.index):
                width = bar.get_width()
                if width > label_threshold:
                    count = count_df.loc[idx, model] if count_df is not None else np.nan
                    if np.isnan(count):
                        label = f"{width * 100:.1f}%"
                    else:
                        label = f"{width * 100:.1f}%\n(n={int(count)})"
                    x = bar.get_x() + width / 2
                    y = bar.get_y() + bar.get_height() / 2
                    ax.text(
                        x,
                        y,
                        label,
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="black",
                        linespacing=0.9,
                    )

    # ax.set_xlabel("Proportion")
    # ax.set_ylabel(group_col)
    # ax.set_title("Best Model Proportion by %s" % group_col)
    handles = [mpatches.Patch(color=colors[model], label=model) for model in models_sorted]
    ax.legend(
        handles=handles,
        title="",
        bbox_to_anchor=(0.5, 1.08),
        loc="upper center",
        ncol=len(models_sorted),
        frameon=False,
    )
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 5))
    ax.set_xticklabels([f"{int(x * 100)}" for x in np.linspace(0, 1, 5)])
    apply_nature_bar_axis_style(
        ax,
        xlabel="Percentage of subjects (%)",
        ylabel="",
        label_fontsize=17,
        tick_fontsize=14,
        grid_axis="x",
    )
    plt.tight_layout()
    return ax

def plot_par_summary(author_summary, palette='viridis', title=None):
    """
    Create a bar plot showing mean PAR values by author_year with subject count labels.
    
    Parameters:
    author_summary (pd.DataFrame): Output from calculate_par_metrics function
    
    Returns:
    matplotlib.figure.Figure: The created figure object
    """

    if author_summary.empty or 'mean_PAR' not in author_summary.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data to display', horizontalalignment='center', 
                verticalalignment='center', transform=ax.transAxes, fontsize=14)
        # ax.set_title(title)
        return fig
    
    color = sns.color_palette(palette, n_colors=len(author_summary))
    fig, ax = plt.subplots(figsize=(max(6, len(author_summary) * 0.4), 4))
    
    # Extract data
    x_pos = range(len(author_summary))
    mean_par_values = author_summary['mean_PAR'].values
    n_subjects = author_summary['n_subjects'].values
    
    # Create bars
    bars = ax.bar(
        x_pos,
        mean_par_values,
        color=color,
        edgecolor='black',
        alpha=0.8,
        width=0.6
    )
    
    # Add subject count labels on top of bars
    for i, (par_val, n_subj) in enumerate(zip(mean_par_values, n_subjects)):
        ax.text(i, par_val + max(mean_par_values) * 0.02, 
                f'n={n_subj}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Styling
    ax.set_ylabel('Mean PAR', fontsize=12)
    ax.set_xlabel('', fontsize=12)
    # ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(mean_par_values) * 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(author_summary.index, rotation=30, ha='right')
    
    plt.tight_layout()
    return fig

def calculate_par_row(group, par_threshold=1.0):
    """
    Calculate PAR and identify the consistent model based on a threshold.
    
    Logic for Consistent Model:
    1. If PAR >= par_threshold: Identify the most frequent model (Mode).
    2. If there is a tie for the top spot (e.g., 2 wins for A, 2 wins for B), 
       return None (ambiguous).
    3. Otherwise, return the most frequent model.
    """
    models = group['winner_model'].to_numpy()
    k = len(models)
    
    if k < 2:
        return pd.Series({'PAR': np.nan, 'consistent_model': None})
    
    # 1. Calculate Agreement (Vectorized)
    # Compare every element with every other element
    same_pairs = (models[:, None] == models[None, :]).sum() - k
    same_pairs //= 2
    n_pairs = k * (k - 1) // 2
    par = same_pairs / n_pairs
    
    consistent_model = None

    # 2. Determine Consistent Model based on Threshold
    if par >= par_threshold:
        # Get value counts to find the mode
        counts = pd.Series(models).value_counts()
        
        if len(counts) == 0:
            consistent_model = None
        elif len(counts) == 1:
            # Only one unique model exists (PAR must be 1.0)
            consistent_model = counts.index[0]
        else:
            # Check for ties in the top position
            top_two = counts.iloc[:2]
            if top_two.iloc[0] > top_two.iloc[1]:
                # Unique winner
                consistent_model = top_two.index[0]
            else:
                # Tie (e.g., A:2, B:2) -> Ambiguous, no consistent model
                consistent_model = None
                
    return pd.Series({'PAR': par, 'consistent_model': consistent_model})

def calculate_par_metrics(df, group_col="author_year", another_group=None, par_threshold=1.0):
    """
    Calculate PAR metrics with a customizable consistency threshold.
    
    Parameters
    ----------
    par_threshold : float, default 1.0
        The minimum PAR value required to assign a 'consistent_model'.
        If PAR >= threshold, the most frequent model is chosen.
    """
    # 1. Calculate PAR for each subject
    group_keys = [group_col, 'subject_id']
    if another_group and another_group in df.columns:
        group_keys.append(another_group)

    # Apply calculation subject-wise, passing the threshold
    # Note: We pass par_threshold as a kwarg to apply
    df_par = (
        df.groupby(group_keys)
        .apply(calculate_par_row, par_threshold=par_threshold)
        .reset_index()
        .dropna(subset=['PAR']) 
    )

    total_subjects_global = len(df_par)

    # 2. Generate Author Summary
    author_summary = (
        df_par.groupby(group_col)
        .agg(
            total_subjects=('subject_id', 'count'),
            n_consistent=('consistent_model', 'count'),  # Renamed from n_perfect
            mean_PAR=('PAR', 'mean'),
            prop_zero=('PAR', lambda x: (x == 0.0).mean())
        )
        .assign(
            # Consistency Rate based on the threshold provided
            percent_consistent=lambda x: (x['n_consistent'] / x['total_subjects'] * 100).round(2)
        )
        .reset_index()
        .sort_values('mean_PAR', ascending=False)
    )

    # 3. Generate Model Stability (Global)
    # Counts based on the new definition of consistency
    model_counts = df_par['consistent_model'].value_counts().reset_index()
    model_counts.columns = ['winner_model', 'n_subj']
    
    model_stability_df = model_counts.assign(
        total_subjects=total_subjects_global,
        percentage=lambda x: (x['n_subj'] / total_subjects_global * 100).round(2)
    )

    return df_par, author_summary, model_stability_df

def plot_bar_with_text(
    df, 
    x_col='winner_model', 
    y_col='consistent_subjects_percentage', 
    count_col='consistent_subjects',
    models_sorted=["DSTP","DMC","SSP","DDM"],
    ax=None, 
    total_subjects=None, 
    palette="viridis", 
    edge_color=None,
    alpha=1,
    gap=0,
    title="", 
    figsize=(8, 5),
    ylabel=None,
    xlabel=None,
    show_percent=True,
    show_count=True,
    show_n_prefix=True,
):
    """
    Create a bar plot showing model stability using Seaborn.
    
    Parameters:
    ----------
    df : pd.DataFrame
        Data containing model stability metrics.
    x_col : str
        Column name for the x-axis (categorical labels).
    y_col : str
        Column name for the y-axis (height of bars, usually percentage).
    count_col : str
        Column name for the sample count (for annotation), optional.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, a new figure is created.
    total_subjects : int, optional
        Total number of subjects to display as text annotation.
    palette : str or list
        Color palette for Seaborn.
    title : str
        Title for the plot.
    ylabel : str
        Label for the y-axis.
    show_percent : bool
        Whether to show the percentage text on top of bars.
    show_count : bool
        Whether to show the count (n=...) text on top of bars.
    
    Returns:
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.
    """
    
    # 1. Handle Ax and Figure creation
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # 2. Handle Empty Data
    if df.empty or y_col not in df.columns:
        ax.text(0.5, 0.5, 'No data to display', 
                horizontalalignment='center', 
                verticalalignment='center', 
                transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        return ax

    # 3. Plotting with Seaborn
    # explicitly setting order avoids mismatched labels if df isn't sorted
    sns.barplot(
        data=df, 
        x=x_col, 
        y=y_col, 
        ax=ax, 
        palette=palette, 
        edgecolor='black', 
        alpha=alpha,
        width=0.6,
        gap=gap,
        hue=x_col, # Assign x to hue to avoid deprecation warning in newer Seaborn
        legend=False
    )

    if edge_color is not None:
        if isinstance(edge_color, dict):
            # Seaborn creates one 'container' per hue level (model)
            for i, container in enumerate(ax.containers):
                # Get the model name associated with this hue level
                model_name = models_sorted[i]
                ec = edge_color.get(model_name, 'black')
                # Apply the color to all bars in this container
                plt.setp(container.patches, edgecolor=ec, linewidth=1.5)
        else:
            # If it's just a single string color, apply to everything
            for container in ax.containers:
                plt.setp(container.patches, edgecolor=edge_color, linewidth=1.5)

    # 4. Add dynamic labels on top of bars
    # We iterate through the patches (bars) drawn by seaborn
    # We assume the order of bars matches the order of the dataframe rows
    y_max = df[y_col].max()
    
    for i, bar in enumerate(ax.patches):
        # Get value from the bar itself (height)
        val = bar.get_height()
        
        # Build the label string
        label_parts = []
        if show_percent:
            label_parts.append(f'{val:.1f}%')
        
        # Access count data safely if requested
        if show_count and count_col in df.columns:
            # We access the dataframe row corresponding to the bar
            count_val = df.iloc[i][count_col]
            if show_n_prefix:
                label_parts.append(f'(n={count_val})')
            else:
                label_parts.append(f'({count_val})')
            
        label_text = "\n".join(label_parts)
        
        # Place text if there is any to place
        if label_text:
            ax.text(
                bar.get_x() + bar.get_width() / 2,  # X center of bar
                val + y_max * 0.03,                 # Slightly above the bar
                label_text, 
                ha='center', 
                va='bottom', 
                fontsize=12, 
                fontweight='bold'
            )

    # 5. Styling
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=13) # Simple auto-format for x-label
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Set Y-limit with some headroom for text
    ax.set_ylim(0, y_max * 1.25)

    # 6. Total Subjects Annotation
    if total_subjects is not None:
        ax.text(0.98, 0.95, f"Total Subjects: {total_subjects}", 
                transform=ax.transAxes, 
                fontsize=12, 
                ha='right', 
                va='top',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    return ax

def plot_subject_task_heatmap(matrix, models, ax=None, figsize=(12, 4)):
    """
    Plot a heatmap showing the best model for each subject-task pair.
    The x-axis is subject index, and the y-axis is task name.

    Parameters
    ----------
    matrix : pd.DataFrame
        DataFrame with subject_id as index, task_name as columns, and model names as values.
    models : list or None
        List of all possible model names. If None, inferred from matrix values.
    ax : matplotlib.axes.Axes or None
        Axes to plot on. If None, a new figure and axes will be created.
    figsize : tuple
        Size of the figure if ax is None.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the heatmap.
    """

    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        matrix,
        cmap="tab10",
        ax=ax,
        cbar=False,
        linewidths=0.5,
        linecolor="gray",
        xticklabels=True,
        yticklabels=True,
    )

    ax.set_xticks([])
    # ax.set_title('Task-Subject Best Model Heatmap')
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Create legend for model colors
    legend_elements = [
        Patch(facecolor=plt.cm.tab10(i / max(1, len(models) - 1)), label=model)
        for i, model in enumerate(models)
    ]
    ax.legend(
        handles=legend_elements,
        title="Model",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    return ax


def plot_model_correlation_heatmap(df, ax=None, figsize=(8, 6)):
    """
    Computes the Pearson correlation matrix of the input DataFrame and visualizes it
    as a heatmap showing only the lower triangle (including diagonal).

    Parameters:
    -----------
    df : pandas.DataFrame
        Input data where each row represents a subject and each column represents a task
        (e.g., 'flanker', 'simon', 'stroop'). Values should be numerical.
    ax : matplotlib.axes.Axes, optional
        The Axes object to draw the plot onto. If None, creates a new figure and axes.
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (8, 6).
        Only used if ax is None.

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The Axes object containing the heatmap visualization.
    """
    # 1. Compute Pearson correlation matrix
    corr = df.corr().iloc[1:, :-1]

    # 2. Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    # 3. Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        created_fig = False

    # 4. Define color palette
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # 5. Plot heatmap
    sns.heatmap(
        corr,
        mask=mask,  # Apply mask to hide upper triangle
        cmap=cmap,  # Use diverging color palette
        vmax=0.3,  # Set maximum value for color scale
        center=0,  # Center color scale at 0
        square=True,  # Force square cells
        linewidths=0.5,  # Set width of lines between cells
        cbar_kws={"shrink": 0.5},  # Adjust color bar size
        annot=True,  # Show correlation values in cells
        fmt=".2f",  # Format for annotation values
        ax=ax,  # Draw on specified axes
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    # 7. Adjust layout if we created the figure
    if created_fig:
        ax.figure.tight_layout()

    return ax


class TaskSessionConsistencyAnalyzer:
    """
    Analyzer for quantifying model selection consistency across sessions within tasks.

    This class calculates various consistency metrics to assess how consistently
    models are selected across different sessions for each task type.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the analyzer with dataset.

        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing columns: task_name, session_id, winner_model
        """
        self.data = data.copy()
        self.tasks = sorted(data["task_name"].unique())
        self.subjects = sorted(data["subject_id"].unique())
        self.sessions = sorted(data["session_id"].unique())
        self.models = sorted(data["winner_model"].unique())

    def calculate_session_consistency_metrics(self):
        """
        Calculate consistency metrics for each session across tasks.

        Returns:
        --------
        pandas.DataFrame
            DataFrame with consistency metrics for each session
        """
        results = []

        for task_i in self.tasks:
            for subject_i in self.subjects:
                subject_data = subject_data = self.data[
                    (self.data["task_name"] == task_i)
                    & (self.data["subject_id"] == subject_i)
                ]
                if subject_data.shape[0] == 0:
                    continue

                models = subject_data["winner_model"].tolist()
                if len(models) == 0:
                    continue

                # Count model occurrences
                model_counts = Counter(models)
                total_sessions = len(models)

                # 1. Shannon entropy
                probs = np.array(list(model_counts.values())) / total_sessions
                shannon_entropy = entropy(probs, base=2)

                # 2. Most frequent model ratio
                most_common_count = max(model_counts.values())
                most_common_ratio = most_common_count / total_sessions

                # 3. Number of unique models
                unique_models = len(model_counts)

                # 4. Consistency score
                max_entropy = np.log2(len(self.models)) if len(self.models) > 1 else 1
                consistency_score = (
                    1 - (shannon_entropy / max_entropy) if max_entropy > 0 else 1
                )

                # 6. Most common model
                most_common_model = model_counts.most_common(1)

                results.append(
                    {
                        "task_name": task_i,
                        "subject_id": subject_i,
                        "total_sessions": total_sessions,
                        "unique_models": unique_models,
                        "shannon_entropy": shannon_entropy,
                        "consistency_score": consistency_score,
                        "most_common_ratio": most_common_ratio,
                        "model_distribution": dict(model_counts),
                        "most_common_model": np.nan
                        if most_common_model[0][1] == 1
                        else most_common_model[0][0],
                    }
                )

        return pd.DataFrame(results)

    def recode_matrix(self, matrix):
        models = self.models
        model_to_num = {model: i for i, model in enumerate(models)}

        # Convert model names to numeric codes for coloring
        numeric_matrix = matrix.applymap(
            lambda x: model_to_num.get(x, -1) if pd.notna(x) else np.nan
        )

        return numeric_matrix

    def create_subject_session_matrix(self):
        pivot_table = self.data.pivot_table(
            index=["subject_id", "task_name"],
            columns="session_id",
            values="winner_model",
            aggfunc="first",
        ).reset_index()

        return pivot_table

    def generate_summary(self, metrics_df):
        unique_models = metrics_df["unique_models"].value_counts(normalize=True)

        return {
            "mean_consistency": metrics_df["consistency_score"].mean(),
            "mean_entropy": metrics_df["shannon_entropy"].mean(),
            "n_win_in_tree_task": unique_models.get(1, 0),
            "n_win_in_two_task": unique_models.get(2, 0),
        }

    def calculate_icc_by_task(self, df, task_col="task_name", subj_col="subject_id"):
        """
        Calculate ICC(2,1) for each task_name using pingouin library.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing task_name, subject_id, and session columns (s1, s2, etc.)
        task_col : str, optional
            Column name for task grouping (default: 'task_name')
        subj_col : str, optional
            Column name for subject identifier (default: 'subject_id')

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: task_name, ICC, n_sessions
            ICC values are ICC(2,1) as calculated by pingouin library
        """
        import pingouin as pg

        # Automatically identify session columns
        session_cols = [
            col for col in df.columns if col not in [task_col, subj_col, "session_id"]
        ]
        results = []

        for task, group in df.groupby(task_col):
            # Convert to long format for pingouin
            data_long = group.melt(
                id_vars=[subj_col],
                value_vars=session_cols,
                var_name="session",
                value_name="value",
            ).rename(columns={subj_col: "subject"})

            # Calculate ICC(2,1) using pingouin
            icc = pg.intraclass_corr(
                data=data_long,
                targets="subject",
                raters="session",
                ratings="value",
                nan_policy="omit",
            )

            # Extract ICC2 (ICC(2,1)) value
            icc2_value = icc[icc["Type"] == "ICC2"]["ICC"].iloc[0]

            results.append(
                {task_col: task, "ICC": icc2_value, "n_sessions": len(session_cols)}
            )

        return pd.DataFrame(results)


def plot_model_comparison_heatmaps(analyzer, matrix, figsize=(5, 4), **kwargs):
    """
    Plot comparison heatmaps for model performance across subjects and tasks.

    For each task:
        - Top row: Correlation heatmap between models
        - Bottom rows: Subject-task heatmap showing winner model per subject

    Parameters:
    -----------
    analyzer : object
        An object containing the following attributes:
            - tasks : list of str
                List of task names
            - models : list of str
                List of model names
            - recode_matrix : function
                Function to process the matrix for plotting

    matrix : pd.DataFrame
        DataFrame containing the data with at least the following columns:
            - 'task_name'
            - 'subject_id'
            - Model-related columns (e.g., metric values)

    plot_model_correlation_heatmap : function
        A plotting function that takes a DataFrame and an Axes object to draw the correlation heatmap.

    plot_subject_task_heatmap : function
        A plotting function that takes a transposed DataFrame, model names, and an Axes object
        to draw the subject-task heatmap.

    Returns:
    --------
    None
        Displays the plot.
    """

    n_task = len(analyzer.tasks)

    # Create figure and GridSpec layout
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        n_task + 1, n_task, **kwargs
    )  # n_task+1 rows, n_task columns

    numeric_matrix = (
        matrix.set_index("subject_id")
        .groupby("task_name")
        .apply(analyzer.recode_matrix, include_groups=False)
        .reset_index()
    )

    for i, task_i in enumerate(analyzer.tasks):
        # Filter matrix for current task
        task_i_numeric_matrix = (
            numeric_matrix[numeric_matrix["task_name"] == task_i]
            .drop("task_name", axis=1)
            .set_index("subject_id")
        )

        # Top row: Correlation heatmap
        ax1 = fig.add_subplot(gs[0, i])
        plot_model_correlation_heatmap(task_i_numeric_matrix, ax=ax1)
        ax1.set_title(task_i)

        # Bottom rows: Subject-task heatmap (spans all columns for this task column)
        ax2 = fig.add_subplot(gs[i + 1, :])
        plot_subject_task_heatmap(task_i_numeric_matrix.T, analyzer.models, ax=ax2)
        ax2.set_title(task_i)

        # Remove legend on all but the first subplot
        if i != 0:
            ax2.get_legend().remove()

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


def merge_icc_results(selected_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Merge ICC dataframes from all author_years in selected_results and add author_years column.

    Parameters
    ----------
    selected_results : Dict[str, Dict[str, Any]]
        Dictionary containing results for different studies, where each study has an 'icc' key
        with a DataFrame containing task_name, ICC, and n_sessions columns.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with columns: task_name, ICC, n_sessions, author_years

    Example
    -------
    >>> selected_results = {
    ...     "hedge2018": {"icc": pd.DataFrame({
    ...         "task_name": ["flanker", "stroop"],
    ...         "ICC": [-0.116614, 0.032927],
    ...         "n_sessions": [2, 2]
    ...     })},
    ...     "lee2025": {"icc": pd.DataFrame({
    ...         "task_name": ["flanker", "stroop"],
    ...         "ICC": [-0.116614, 0.032927],
    ...         "n_sessions": [2, 2]
    ...     })}
    ... }
    >>> merged_df = merge_icc_results(selected_results)
    >>> print(merged_df)
      task_name       ICC  n_sessions author_years
    0   flanker -0.116614           2    hedge2018
    1    stroop  0.032927           2    hedge2018
    2   flanker -0.116614           2     lee2025
    3    stroop  0.032927           2     lee2025
    """
    merged_dfs = []

    # Iterate through each study in selected_results
    for author_year, data in selected_results.items():
        # Check if 'icc' key exists in the study data
        if "icc" in data:
            # Copy the ICC dataframe and add author_years column
            df = data["icc"].copy()
            df["author_years"] = author_year
            merged_dfs.append(df)

    # Concatenate all dataframes if any exist
    if merged_dfs:
        merged_df = pd.concat(merged_dfs, ignore_index=True)
        return merged_df
    else:
        # Return empty dataframe with expected columns if no data
        return pd.DataFrame(columns=["task_name", "ICC", "n_sessions", "author_years"])


def plot_icc_comparison(
    merged_icc_df: pd.DataFrame,
    x="task_name",
    hue="author_years",
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (5, 4),
    title: str = "ICC of multiple tests \n across task and study",
    palette: str = "Set2",
) -> plt.Axes:
    """
    Plot ICC comparison bar chart with task_name on x-axis and author_years as legend.

    Parameters
    ----------
    merged_icc_df : pd.DataFrame
        Merged DataFrame containing task_name, ICC, and author_years columns
    ax : plt.Axes, optional
        Matplotlib axes object to plot on. If None, a new figure and axes will be created.
    figsize : tuple, optional
        Figure size as (width, height), by default (10, 6). Only used if ax is None.
    title : str, optional
        Chart title, by default 'ICC Values by Task and Study'
    palette : str, optional
        Color palette for the bars, by default 'Set2'

    Returns
    -------
    plt.Axes
        Matplotlib axes object with the plot

    Example
    -------
    >>> # Assuming merged_icc_df is already created using merge_icc_results
    >>> ax = plot_icc_comparison(merged_icc_df)
    >>> plt.show()
    >>>
    >>> # Plot on existing axes
    >>> fig, ax = plt.subplots(figsize=(12, 8))
    >>> ax = plot_icc_comparison(merged_icc_df, ax=ax, title='Custom Title')
    >>> plt.show()
    >>>
    >>> # Custom parameters
    >>> ax = plot_icc_comparison(
    ...     merged_icc_df,
    ...     figsize=(12, 8),
    ...     title='Interrater Reliability Comparison',
    ...     palette='viridis'
    ... )
    >>> plt.show()
    """
    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Create barplot using seaborn
    bar_plot = sns.barplot(
        data=merged_icc_df, x=x, hue=hue, y="ICC", palette=palette, ax=ax
    )

    # Add value labels on bars
    for container in bar_plot.containers:
        bar_plot.bar_label(container, fmt="%.3f", padding=3, fontsize=8)

    # Set chart properties
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("", fontsize=12)
    ax.set_ylabel("ICC", fontsize=12)
    ax.legend(title="", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.7)

    # Set appropriate y-axis limits
    if not merged_icc_df.empty:
        y_min = merged_icc_df["ICC"].min() - 0.05
        y_max = merged_icc_df["ICC"].max() + 0.05
        ax.set_ylim(y_min, y_max)

    # Adjust layout to prevent legend cutoff if we created the figure
    if ax.figure is not None:
        ax.figure.tight_layout()

    sns.despine()
    return ax

def plot_model_proportions_by_task(data:pd.DataFrame, model_col='winner_model', group_col=None, 
                                   ax=None, figsize=(10, 6), xlabel=None, ylabel='Percentage %',
                                   palette='viridis',
                                   xtick_fontsize=12,
                                   ylabel_fontsize=12,
                                   edge_color=None,
                                   alpha=1,
                                   gap=0.18,
                                   models_sorted=["DDM","SSP","DMC","DSTP"],
                                   rotate_x_labels=False):
    """
    Plot model proportions by task as a grouped bar chart.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing at least group_col and model_col columns
    group_col : str, optional
        Column name for task grouping, by default None
        If None, plot model proportions across entire dataset
    model_col : str, optional
        Column name for model identification, by default 'winner_model'
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates new figure and axes.
    figsize : tuple, optional
        Figure size as (width, height), by default (10, 6)
    xlabel : str, optional
        Label for x-axis. If None, uses group_col or model_col value.
    ylabel : str, optional
        Label for y-axis, by default 'Percentage'
    edge_color : str, optional
        Color for the bar borders, by default None
    rotate_x_labels : bool, optional
        Whether to rotate x-axis labels by 90 degrees, by default False
        
    Returns
    -------
    matplotlib.axes.Axes
        Axes object with the plot
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # sns.set_palette(palette=palette)
    
    # Create plot if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    if group_col is not None:
        # Calculate proportions by group
        group_model_proportions = data.groupby([group_col, model_col]).size().reset_index(name='count')
        group_totals = data.groupby(group_col).size().reset_index(name='group_total')
        merged_df = pd.merge(group_model_proportions, group_totals, on=group_col)
        merged_df['percentage'] = (merged_df['count'] / merged_df['group_total'] * 100).round(2)
        
        # Generate grouped bar plot
        sns.barplot(data=merged_df, x=group_col, y='percentage', hue=model_col, 
                    hue_order=models_sorted, ax=ax, palette=palette, alpha=alpha, gap=gap)
        
        # Set labels
        ax.set_xlabel(xlabel if xlabel is not None else group_col)
        ax.legend(title='')
    else:
        # Calculate overall proportions
        model_proportions = data[model_col].value_counts(normalize=True).reset_index()
        model_proportions.columns = [model_col, 'percentage']
        model_proportions['percentage'] = (model_proportions['percentage'] * 100).round(2)
        model_proportions_sorted = model_proportions.set_index(model_col).reindex(models_sorted).reset_index()

        # Generate simple bar plot
        sns.barplot(data=model_proportions_sorted, x=model_col, y='percentage', 
                    hue=model_col, ax=ax, palette=palette, alpha=alpha, gap=gap)
        
        # Set labels
        ax.set_xlabel(xlabel if xlabel is not None else model_col)
    
    # 2. Apply edge_color manually after the bars are created
    if edge_color is not None:
        if isinstance(edge_color, dict):
            # Seaborn creates one 'container' per hue level (model)
            for i, container in enumerate(ax.containers):
                # Get the model name associated with this hue level
                model_name = models_sorted[i]
                ec = edge_color.get(model_name, 'black')
                # Apply the color to all bars in this container
                plt.setp(container.patches, edgecolor=ec, linewidth=1.5)
        else:
            # If it's just a single string color, apply to everything
            for container in ax.containers:
                plt.setp(container.patches, edgecolor=edge_color, linewidth=1.5)
                
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    
    ax.tick_params(axis='x', labelsize=xtick_fontsize)
    # Rotate x labels if requested
    if rotate_x_labels:
        ax.tick_params(axis='x', rotation=rotate_x_labels)
    
    return ax

def plot_bars(
    df,
    x,
    dv,
    group=None,
    rotate_xtick=60,
    title="",
    y_label="",
    legend_loc="right",
    palette="viridis",
    ax=None,
):
    """
    Plot bar chart on a given or new matplotlib Axes.

    Parameters:
    ----------
    df : pd.DataFrame
        Data to be plotted.
    x : str
        Column name for x-axis (e.g., parameter names).
    dv : str
        Column name for dependent variable (e.g., ICC values).
    group : str, optional
        Column name used for grouping (optional, used for coloring bars).
    rotate_xtick : int
        Rotation degree of xtick labels.
    title : str
        Title of the subplot.
    y_label : str
        Label for y-axis.
    legend_loc : str
        Legend location ('right' or other valid location).
    palette : str or list
        Color palette name or list of colors.
    ax : matplotlib.axes.Axes, optional
        Predefined axes to draw on. If None, a new one is created.

    Returns:
    -------
    matplotlib.axes.Axes
        The Axes object with the plot.
    """
    df = df.copy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.5), 5))

    if group is not None and group in df.columns:
        df = df[df[group].notna()]
        groups = df[group].astype(str)
        unique_groups = groups.unique()
        palette = sns.color_palette(palette, n_colors=len(unique_groups))
        color_map = {grp: palette[i] for i, grp in enumerate(unique_groups)}
        bar_colors = groups.map(color_map)

    else:
        bar_colors = None

    labels = df[x].values
    bars = ax.bar(range(len(labels)), df[dv], color=bar_colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=rotate_xtick, ha="right")
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_ylim(0, 1)
    ax.axhline(0.4, linestyle='--', color='gray', alpha=0.5)

    # Add legend if grouped
    if group is not None and group in df.columns:
        handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[grp]) for grp in unique_groups]
        if legend_loc == "right":
            ax.legend(
                handles,
                unique_groups,
                title="",
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                borderaxespad=0,
            )
        else:
            ax.legend(handles, unique_groups, title="", loc=legend_loc)

    sns.despine(ax=ax)
    return ax

def wrap_parameter(param):
    """
    Wrap parameter name in $...$ format and add {} around parts after underscore.
    
    Parameters
    ----------
    param : str
        Parameter name to wrap
        
    Returns
    -------
    str
        Wrapped parameter name with proper LaTeX formatting
        
    Examples
    --------
    >>> wrap_parameter('rt_avg')
    '$rt_{avg}$'
    >>> wrap_parameter('v_cong|DDM')
    '$v_{cong}|DDM$'
    >>> wrap_parameter('$rt_{avg}$')
    '$rt_{avg}$'
    """
    param_str = str(param)
    
    # If already wrapped in $, return as is
    if param_str.startswith('$') and param_str.endswith('$'):
        return param_str
    
    # Handle underscore formatting
    if '_' in param_str:
        # Split by underscore and wrap the part after underscore in {}
        parts = param_str.split('_', 1)  # Split only on first underscore
        formatted_param = f'{parts[0]}_{{{parts[1]}}}'
        return f'${formatted_param}$'
    else:
        # No underscore, simple wrapping
        return f'${param_str}$'

def compare_factor_iccs(icc_df, factor_vars, group_col=None):
    """
    Compare ICC values of synthetic factors and their constituent variables,
    grouped by a specified column (e.g., 'author_year' or 'task_name').

    Parameters:
    - icc_df: pandas.DataFrame
        DataFrame containing ICC results with at least columns: 'parameter', 'ICC', and group_col.
    - group_col: str
        Column name to group by (e.g., 'author_year' or 'task_name').
    - factor_vars: dict
        Dictionary mapping factor names to lists of constituent parameter names.

    Returns:
    - result_df: pandas.DataFrame
        DataFrame with columns: [group_col, 'Factor', 'Component', 'ICC']
        Rows represent each component's ICC within each factor and group.
    """
    # Initialize list to store results
    rows = []

    # Get unique groups
    if group_col is None:
        group_col_name = 'Group'
        unique_groups = ['All']
        icc_df = icc_df.copy()
        icc_df[group_col_name] = 'All'
    else:
        group_col_name = group_col
        unique_groups = icc_df[group_col].dropna().unique()

    # For each group
    for group in unique_groups:
        group_data = icc_df[icc_df[group_col_name] == group]

        # For each factor
        for factor, components in factor_vars.items():
            for comp in components:

                # Look up ICC for this component in current group
                match = group_data[group_data['parameter'] == comp]

                if not match.empty:
                    icc_val = match['ICC'].mean()
                else:
                    icc_val = None  # Component not found in this group

                # Append row
                rows.append({
                    group_col_name: group,
                    'Factor': factor,
                    'Component': comp,
                    'ICC': icc_val
                })

    result_df = (
        pd.DataFrame(rows)
        .sort_values(by=[group_col_name, 'Factor', 'ICC'], ascending=[True, True, False])
        .reset_index(drop=True)
    )

    return result_df


def rank_models_by_metric(df, metric):
    """
    Ranks models within each task_id and subject_id based on a specified metric.
    
    Parameters:
    df (pd.DataFrame): The input long-format dataframe.
    metric (str): The column name to use for ranking (e.g., 'aBIC', 'chi_square', 'RMSE').
    
    Returns:
    pd.DataFrame: Dataframe with an additional 'model_rank' column.
    """
    
    # Check if the specified metric exists in the dataframe
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in dataframe columns.")

    df = df.copy()
    df = df.dropna(subset=[metric])
    # Group by task_id and subject_id, then rank the metric
    # ascending=True ensures the lowest value gets the rank 1
    # method='min' or 'average' can be adjusted if ties occur; 
    # here we use default which provides distinct ranks if possible or handles ties.
    df['model_rank'] = df.groupby(['task_id', 'subject_id'])[metric].rank(ascending=True, method='min').astype(int)
    
    return df
 
def summarize_model_performance(df, group_col, metric):
    """
    Summarizes group-level performance:
    1. Identifies the most frequent rank-1 model (Group Best).
    2. Calculates mean/std for Group Best, Subject Best (rank 1), and Subject Worst (rank 4).
    
    Parameters:
    df (pd.DataFrame): Input dataframe with 'model_rank' and 'model'.
    group_col (str): Column to group by (e.g., 'task_id').
    metric (str): The column name to calculate statistics for.
    
    Returns:
    pd.DataFrame: Simplified summary table.
    """
    
    # --- 1. Identify the Group Best Model (Most frequent rank 1) ---
    # Filter for models that are ranked #1 for each subject
    rank1_df = df[df['model_rank'] == 1].copy()
    
    # Find the model name that appears most often per group
    group_best_model_map = (
        rank1_df.groupby([group_col, 'model'])
        .size()
        .reset_index(name='count')
        .sort_values([group_col, 'count'], ascending=[True, False])
        .drop_duplicates(subset=[group_col])
        .set_index(group_col)['model']
    )
    
    # --- 2. Helper for Stats Calculation ---
    def get_stats(data, prefix):
        return (
            data.groupby(group_col)[metric]
            .agg(['mean', 'std'])
            .rename(columns={'mean': f'{prefix}_mean', 'std': f'{prefix}_std'})
        )

    # --- 3. Compute Stats for different categories ---
    
    # A. Subject Best (Whichever model is rank 1 for each person)
    subject_best_stats = get_stats(rank1_df, 'subject_best')
    
    # B. Subject Worst (Whichever model is rank 4 for each person)
    rank4_df = df[df['model_rank'] == 4].copy()
    subject_worst_stats = get_stats(rank4_df, 'subject_worst')
    
    # C. Group Best (Performance of the specific group-winner model across all subjects)
    # Map the group best model back to the original dataframe rows
    df_with_winner = df.copy()
    df_with_winner['is_group_best'] = df_with_winner.apply(
        lambda row: row['model'] == group_best_model_map.get(row[group_col]), axis=1
    )
    group_best_stats = get_stats(df_with_winner[df_with_winner['is_group_best']], 'group_best')

    # --- 4. Assemble the Final Table ---
    summary = pd.DataFrame(group_best_model_map).rename(columns={'model': 'best_model_name'})
    summary = (
        summary.join(group_best_stats)
               .join(subject_best_stats)
               .join(subject_worst_stats)
               .reset_index()
    )
    
    return summary
  
def plot_summary_results(summary_df, group_col, colors, ax=None, 
                         figsize=(10, 6), xlabel='', 
                         ylabel='', title=None, markersize=8, 
                         capsize=5, alpha=0.7, linestyle='None'):
    """
    Plots the mean values and standard deviations for Group Best, Subject Best, 
    and Subject Worst models across different groups.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        The output from the summarize_model_performance function.
    group_col : str
        The column used for grouping (e.g., 'task_id').
    palette : str, optional
        Seaborn color palette name, by default "viridis".
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates new figure and axes.
    figsize : tuple, optional
        Figure size as (width, height), by default (10, 6).
    xlabel : str, optional
        Label for x-axis, by default 'Performance Score'.
    ylabel : str, optional
        Label for y-axis, by default 'Group'.
    title : str, optional
        Title for the plot, by default None.
    markersize : int, optional
        Size of markers, by default 8.
    capsize : int, optional
        Size of error bar caps, by default 5.
    alpha : float, optional
        Transparency of error bars, by default 0.7.
    linestyle : str, optional
        Line style for connecting points, by default 'None'.
        
    Returns
    -------
    matplotlib.axes.Axes
        Axes object with the plot.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set the visual style
    sns.set_theme(style="whitegrid")
    
    # Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # 1. Prepare data for plotting (Convert from wide to long format)
    # Define the categories we want to plot
    categories = ['group_best', 'subject_best', 'subject_worst']
    
    plot_data_list = []
    for cat in categories:
        temp_df = summary_df[[group_col, f'{cat}_mean', f'{cat}_std']].copy()
        temp_df.columns = [group_col, 'mean', 'std']
        temp_df['type'] = cat
        plot_data_list.append(temp_df)
    
    plot_df = pd.concat(plot_data_list, ignore_index=True)
    
    # 2. Set up colors and markers
    markers = {"group_best": "o", "subject_best": "D", "subject_worst": "X"}
    
    # 3. Create the plot
    # We use errorbar to show both mean and standard deviation
    for i, cat in enumerate(categories):
        subset = plot_df[plot_df['type'] == cat]
        
        # Plotting the points with error bars
        ax.errorbar(
            x=subset['mean'], 
            y=subset[group_col], 
            xerr=subset['std'], 
            fmt=markers[cat], 
            color=colors[i], 
            label=cat.replace('_', ' ').title(),
            capsize=capsize,
            markersize=markersize,
            alpha=alpha,
            linestyle=linestyle
        )
    
    # 4. Add labels and legend
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14, pad=12)
    
    ax.legend(title='')
    
    # 5. Adjust layout
    plt.tight_layout()
    
    return ax

def add_subplot_label(ax, label, x_offset=-0.1, y_offset=1.05, **kwargs):
    kwargs.setdefault("fontsize", 16)
    kwargs.setdefault("fontweight", "bold")
    ax.text(x_offset, y_offset, label, transform=ax.transAxes, 
            va='top', ha='right', **kwargs)
    return ax

def plot_stacked_consistency(
    df_par, 
    group_col="author_year", 
    palette="viridis", 
    figsize=(8, 6), 
    custom_colors=None,  
    ylabel="Percentage of Subjects (%)",
    ax=None, 
    alpha=0.8,
    models_sorted=["DDM","SSP","DMC","DSTP"], 
    save_path=None,
    rotate_xticks=None,
    label_threshold: float = 0,
    edge_color=None,
    show_legend=True,
):
    """
    Create a stacked bar chart showing the proportion of consistent models vs inconsistent subjects.

    Parameters
    ----------
    df_par : pd.DataFrame
        Output from calculate_par_metrics.
    group_col : str
        X-axis grouping column.
    palette : str
        Color palette for the plot.
    figsize : tuple
        Figure size (width, height).
    save_path : str or None
        Path to save the figure. If None, figure is not saved.
    rotate_xticks : 
        Whether to rotate x-axis tick labels for better readability.
    """
    
    # 1. Handle Ax and Figure creation
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    
    # 1. Prepare Data for Plotting
    # Fill NaN in 'consistent_model' with 'Inconsistent' string for counting
    plot_data = df_par.copy()
    plot_data["consistent_model"] = plot_data["consistent_model"].fillna("Inconsistent")

    # Pivot: Index=Author, Columns=Model, Values=Count
    pivot_df = (
        plot_data.groupby([group_col, "consistent_model"]).size().unstack(fill_value=0)
    )

    # Calculate Total N per author for annotation
    total_n = pivot_df.sum(axis=1)

    # Convert counts to Percentages
    pivot_pct = pivot_df.div(total_n, axis=0) * 100

    # Sort authors (optional: e.g., by proportion of Inconsistent ascending, or by Total N)
    # Here sorting by the sum of "Consistent" models (i.e., 100 - Inconsistent)
    if "Inconsistent" in pivot_pct.columns:
        sort_order = pivot_pct["Inconsistent"].sort_values(ascending=False).index
    else:
        sort_order = pivot_pct.sum(axis=1).sort_values(ascending=False).index

    pivot_pct = pivot_pct.loc[sort_order]
    total_n = total_n.loc[sort_order]

    # 2. Define Colors
    # Get unique models excluding 'Inconsistent'
    models = [c for c in pivot_pct.columns if c != "Inconsistent"]

    if custom_colors:
        # Use provided dictionary
        color_map = custom_colors.copy()
        color_map["Inconsistent"] = "#ffffff" # Ensure white
    else:
        # Generate new palette
        colors = sns.color_palette(palette, n_colors=len(models))
        color_map = dict(zip(models, colors))
        color_map["Inconsistent"] = "#ffffff"

    # Reorder columns: Models first, Inconsistent last
    cols_ordered = models + ["Inconsistent"] if "Inconsistent" in pivot_pct.columns else models
    pivot_pct = pivot_pct[cols_ordered]
    models_sorted.append("Inconsistent")

    # Create color list for the specific columns present in this dataframe
    # Use .get() to avoid errors if a model exists in color_map but not in pivot_pct
    plot_colors = [color_map.get(c, '#333333') for c in pivot_pct.columns]

    # Plot stacked bar without default black edges unless specified
    bars = pivot_pct.plot(
        kind="bar",
        stacked=True,
        color=plot_colors,
        alpha=alpha,
        linewidth=0.8,
        ax=ax,
        width=0.72,
    )

    if edge_color is not None:
        if isinstance(edge_color, dict):
            # The order of containers corresponds to the columns in pivot_pct
            for i, container in enumerate(ax.containers):
                model_name = pivot_pct.columns[i]
                ec = edge_color.get(model_name, 'black')
                # 'Inconsistent' defaults to a light gray or black if not in edge_color dict
                if model_name == 'Inconsistent':
                    ec = "#cccccc"
                import matplotlib.pyplot as plt
                plt.setp(container.patches, edgecolor=ec, linewidth=1.0)
        else:
            for container in ax.containers:
                import matplotlib.pyplot as plt
                plt.setp(container.patches, edgecolor=edge_color, linewidth=1.0)
    else:
        for container in ax.containers:
            import matplotlib.pyplot as plt
            plt.setp(container.patches, edgecolor="black", linewidth=0.8)

    # 4. Add percentage labels on each segment of the stacked bars
    for i, (idx, row) in enumerate(pivot_pct.iterrows()):
        cumulative_height = 0
        for j, value in enumerate(row):
            if value > label_threshold:
                height = value
                ax.text(
                    i,  # x position (bar index)
                    cumulative_height + height / 2,  # y position (middle of segment)
                    f'{value:.1f}%',  # text to show
                    ha='center',
                    va='center',
                    fontsize=9,
                    fontweight='normal',
                )
            cumulative_height += value  # Always increment cumulative height, even if no label

    # 5. Annotations and Styling
    ax.set_ylim(0, 115)  # Extra space for N labels
    ax.set_ylabel(ylabel, fontsize=14)
    
    # Remove title and xlabel as requested
    # ax.set_title("Model Consistency Across Studies", fontsize=16, fontweight="bold")
    # ax.set_xlabel(group_col.replace("_", " ").title(), fontsize=14)
    ax.set_xlabel("", fontsize=14)
    
    # Apply rotation to x-tick labels if requested
    if rotate_xticks is not None:
        plt.xticks(rotation=rotate_xticks, ha="right")
    else:
        plt.xticks(rotation=0)
        
    # Add N labels on top of bars
    for i, author in enumerate(pivot_pct.index):
        n_val = total_n[author]
        # Position: x=i, y=100 (top of stack) + offset
        ax.text(
            i,
            102,
            f"N={n_val}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Customize Legend
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        label_to_handle = dict(zip(labels, handles))
        sorted_handles = [label_to_handle[label] for label in models_sorted if label in label_to_handle]
        
        # Ensure Inconsistent label is clear
        ax.legend(
            sorted_handles,
            models_sorted,
            title="Consistent Model",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=False,
        )
    elif ax.get_legend() is not None:
        ax.get_legend().remove()

    if ax is None:
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="svg", bbox_inches="tight")

    return fig

def plot_par_grid_optimized(df_par, custom_colors=None, pct_threshold=10,alpha=0.8, models_sorted=["DDM","SSP","DMC","DSTP"]):
    """
    Creates a grid of pie charts showing distribution of consistent models by task and author-year.
    
    Parameters:
    - df_par (pd.DataFrame): DataFrame containing columns 'task_name', 'author_year', and 'consistent_model'
    - custom_colors (dict, optional): Dictionary mapping model names to custom colors
    - pct_threshold (float): Minimum percentage threshold for displaying percentage labels on pie charts
    - models_sorted (list): List of model names in the desired order for legend display
    
    Returns:
    - g: Seaborn FacetGrid object containing the plot
    """
    
    # --- 1. Data Preparation ---
    plot_data = df_par.copy()
    inconsistent_label = "Inconsistent"
    
    # Fill NaN values
    plot_data['consistent_model'] = plot_data['consistent_model'].fillna(inconsistent_label)
    
    # Determine model ordering (Inconsistent goes last)
    unique_models = sorted([m for m in plot_data['consistent_model'].unique() if m != inconsistent_label])
    all_categories = unique_models + [inconsistent_label]

    n_counts = plot_data['author_year'].value_counts()

    original_order = sorted(plot_data['author_year'].unique())
    def get_fmt_label(year):
        count = n_counts.get(year, 0)
        return f"{year}\n(N={count})"
    fmt_order = [get_fmt_label(year) for year in original_order]
    label_map = {year: get_fmt_label(year) for year in original_order}
    plot_data['author_year_fmt'] = plot_data['author_year'].map(label_map)

    # --- 2. Color Mapping ---
    # Default viridis palette
    viridis_colors = sns.color_palette("viridis", n_colors=len(unique_models))
    model_color_map = dict(zip(unique_models, viridis_colors))
    
    # Apply custom colors if provided
    if custom_colors:
        for model, color in custom_colors.items():
            if model in model_color_map:
                model_color_map[model] = color
    
    # Force Inconsistent to be white
    final_color_map = model_color_map.copy()
    final_color_map[inconsistent_label] = '#FFFFFF' 

    # --- 3. Helper function: Percentage filter ---
    def filter_autopct(pct):
        return f'{pct:.0f}%' if pct > pct_threshold else ''

    # --- 4. Drawing function (Map Function) ---
    def draw_pie(data, **kwargs):
        # Get current axes
        ax = plt.gca()
        
        # If no data, turn off axes and return (blank space)
        if data.empty:
            ax.axis('off')
            return

        counts = data['consistent_model'].value_counts()
        
        # Even if there's data, if it's all NaN (Inconsistent) and gets filtered out, handle it
        if counts.empty:
            ax.axis('off')
            return

        colors = [final_color_map[cat] for cat in counts.index]
        
        # Draw pie chart
        wedges, texts, autotexts = ax.pie(
            counts, 
            colors=colors, 
            radius=1.3,           # Slightly enlarge
            startangle=90,
            wedgeprops={'edgecolor': 'gray', 'linewidth': 0.8, 'antialiased': True, 'alpha':alpha},
            autopct=filter_autopct, # Use the filtering function above
            textprops={'fontsize': 10, 'color': 'black'} # Percentage text color
        )
        
        # Adjust position of percentage labels (optional: bring them closer to center)
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            # If Inconsistent (white background), text could be set to gray to avoid being too prominent, or keep black
            # Keeping black here for readability

    # --- 5. Create Grid ---
    with sns.plotting_context("notebook", font_scale=1.2):
        g = sns.FacetGrid(
            plot_data, 
            row="task_name", 
            col="author_year_fmt", 
            margin_titles=True,  # Key: show row labels on the right side
            height=1.5, 
            aspect=1,
            sharex=False, 
            sharey=False,
            despine=True # Remove borders
        )

    # Map drawing function
    g.map_dataframe(draw_pie)

    # --- 6. Style adjustments ---
    g.set_titles(col_template="{col_name}", row_template="{row_name}",size=14, fontweight='bold')
    for ax in g.axes[0, :]:  
        ax.title.set_fontsize(12)
    for ax in g.axes[:, -1]:  
        pass
    
    # Clear default axes for each subplot (pie charts don't need rectangular frames)
    for ax in g.axes.flat:
        ax.axis('off')

    # --- 7. Custom Legend (no title) ---
    legend_handles = []
    # Models (Viridis) - ordered according to models_sorted
    for model in models_sorted:
        if model in unique_models:
            legend_handles.append(mpatches.Patch(color=model_color_map[model], label=model,alpha=alpha))
    # Inconsistent (White)
    legend_handles.append(mpatches.Patch(facecolor='white', edgecolor='gray', label=inconsistent_label))

    # Position legend on the right side, vertically
    plt.legend(
        handles=legend_handles, 
        bbox_to_anchor=(0.9, 0.5), # Put at right side, centered vertically
        bbox_transform=g.figure.transFigure,
        loc='center left',
        frameon=False,
        fontsize=10,
        title="Consistent Model"
    )

    # Adjust layout to leave space at right for legend
    g.figure.subplots_adjust(top=0.85, bottom=0.12, right=0.85, hspace=0.2, wspace=0.2)  # Reduced spacing between subplots
    
    return g

def compute_icc_consistency(
    df: pd.DataFrame,
    subject_col: str = 'subject_id',
    contrast_col: str = 'task_name',
    ignore_cols: list = ['task_id', 'author_year']
) -> pd.DataFrame:
    """
    Compute Intraclass Correlation Coefficient (ICC) to assess consistency of parameters across different tasks.

    Parameters:
    ----------
    df : pd.DataFrame
        Input data frame. Each row represents a subject's parameter under a specific task.
    subject_col : str, default='subject_id'
        Column name indicating subject IDs.
    task_col : str, default='task_name'
        Column name indicating task names.
    ignore_cols : list, optional
        List of columns to be ignored in the analysis (e.g., non-parameter columns).

    Returns:
    -------
    pd.DataFrame
        A DataFrame where each row corresponds to a parameter, with ICC results including:
        - type: Type of ICC used (ICC2)
        - ICC: Intraclass correlation coefficient
        - F: F statistic
        - df1, df2: Degrees of freedom
        - pval: P-value
        - CI95%: 95% confidence interval for ICC

    Example:
    --------
    >>> icc_results = compute_icc_consistency(df, ignore_cols=['task_id', 'author_year'])
    >>> print(icc_results[['ICC', 'pval', 'CI95%']])
    """
    if pg is None:
        raise ImportError(
            "pingouin is required for compute_icc_consistency; install pingouin to use this function."
        )

    if ignore_cols is None:
        ignore_cols = []

    # Remove irrelevant columns
    cols_to_use = [col for col in df.columns if col not in ignore_cols + [subject_col, contrast_col]]

    results = {}

    for param in cols_to_use:
        # Long format for ICC analysis
        data_long = df[[subject_col, contrast_col, param]].copy()
        data_long = data_long.rename(columns={param: 'value'})

        # Drop missing values
        data_long.dropna(inplace=True)

        # Ensure there are at least two tasks and multiple subjects
        if len(data_long[contrast_col].unique()) < 2 or len(data_long[subject_col].unique()) < 2:
            continue

        try:
            icc = pg.intraclass_corr(
                data=data_long,
                targets=subject_col,
                raters=contrast_col,
                ratings='value',
                nan_policy='omit'
            )
            icc = icc[icc['Type'] == 'ICC2']
            if not icc.empty:
                results[param] = icc.iloc[0].to_dict()
        except Exception as e:
            print(f"Error computing ICC for {param}: {e}")
            continue

    # Convert to DataFrame
    results_df = pd.DataFrame(results).T
    return results_df

def process_icc_data(
    data, 
    contrast_col, 
    subject_col='subject_id', 
    ignore_cols=None, 
    group_col=None, 
    key_label='author_year', 
    regex_extract=None,
    factor_vars=None
):
    """
    Unified function to compute ICC for dictionary or DataFrame inputs, 
    parse model parameters, and map factors.

    Parameters
    ----------
    data : dict or pd.DataFrame
        - If dict: {study_name: df, ...}. Iterates keys.
        - If DataFrame: Groups by `group_col` before computing.
    contrast_col : str
        The column representing repeated measures (e.g., 'task_name' or 'session_id').
    subject_col : str
        Subject identifier column.
    ignore_cols : list
        Columns to exclude from ICC calculation.
    group_col : str, optional
        Required if input is a DataFrame. The column to group by (e.g., 'task_id').
    key_label : str, optional
        Only for dict input. The name for the key column (default: 'author_year').
    regex_extract : tuple, optional
        Only for DataFrame input. Format: (r'regex_pattern', ['col_name1', 'col_name2']).
    factor_vars : dict, optional
        Dictionary mapping factor names to lists of parameters.
        Example: {"Caution": ["$a|DDM$", "$a|DMC$"]}

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame containing ICC results with parsed columns.
    """
    
    results_list = []

    # --- Scenario 1: Dictionary Input (Cross-Task Consistency) ---
    if isinstance(data, dict):
        for key, df_sub in data.items():
            # Compute ICC using external function
            icc = compute_icc_consistency(
                df=df_sub, 
                subject_col=subject_col, 
                contrast_col=contrast_col, 
                ignore_cols=ignore_cols
            )
            
            # Format to Long
            temp = icc.reset_index()
            # Standardize parameter column name (usually 'index' after reset)
            temp = temp.rename(columns={'index': 'parameter'})
            
            # Add the dictionary key as a column
            temp[key_label] = key
            results_list.append(temp)
            
        final_df = pd.concat(results_list, ignore_index=True)

    # --- Scenario 2: DataFrame Input (Test-Retest / GroupBy) ---
    elif isinstance(data, pd.DataFrame):
        if group_col is None:
            raise ValueError("`group_col` must be provided when input is a DataFrame.")

        # Define apply wrapper
        def _apply_icc(x):
            return compute_icc_consistency(
                df=x, 
                subject_col=subject_col, 
                contrast_col=contrast_col, 
                ignore_cols=ignore_cols
            )

        # GroupBy and Apply
        icc_grouped = data.groupby(group_col).apply(_apply_icc)
        
        # Reset index to flatten
        final_df = icc_grouped.reset_index()
        
        # Standardize parameter column name
        # After reset_index, the parameter index usually becomes 'level_1' or 'index'
        if 'level_1' in final_df.columns:
            final_df = final_df.rename(columns={'level_1': 'parameter'})
        elif 'index' in final_df.columns:
            final_df = final_df.rename(columns={'index': 'parameter'})

        # Regex Extraction for Study/Task info (if provided)
        if regex_extract:
            pattern, new_cols = regex_extract
            extracted = final_df[group_col].astype(str).str.extract(pattern)
            final_df[new_cols] = extracted

    else:
        raise TypeError("Input `data` must be a dict or pd.DataFrame.")

    # --- Post-Processing: Parameter Parsing & Factor Mapping ---
    
    if not final_df.empty and 'parameter' in final_df.columns:
        
        # 1. Parse "$param|Model$" format
        # Regex explanation:
        # ^\$      : Start with a literal '$'
        # (.*?)    : Group 1 - The parameter name (non-greedy)
        # \|       : Literal '|' separator
        # (.*?)    : Group 2 - The model name (non-greedy)
        # \$$      : End with a literal '$'
        pattern = r'^\$(.*?)\|(.*?)\$$'
        
        # Extract returns a DataFrame with two columns (0 and 1) corresponding to groups
        extracted_cols = final_df['parameter'].astype(str).str.extract(pattern)
        
        # Assign new columns
        final_df['param_name'] = extracted_cols[0]
        final_df['model_name'] = extracted_cols[1]
        
        # 2. Map Factors based on `factor_vars` dictionary
        if factor_vars:
            # Invert the dictionary to map {parameter: factor_name}
            # Example input: {"Caution": ["$a|DDM$"]} -> Output: {"$a|DDM$": "Caution"}
            param_to_factor_map = {
                param: factor 
                for factor, params_list in factor_vars.items() 
                for param in params_list
            }
            
            # Map the original 'parameter' column to the new 'factor' column
            final_df['factor'] = final_df['parameter'].map(param_to_factor_map)

    return final_df


def plot_icc_distribution(
    df, 
    x_col='factor', 
    y_col='ICC', 
    hue_col='model_name', 
    col_col='author_year', 
    row_col=None,
    ref_lines=[0.4, 0.7],
    palette='viridis',
    height=2.5,
    aspect=0.8,
    sharey=True
):
    """
    Visualizes ICC distribution using a Boxplot + Stripplot overlay, faceted by Author and Task.

    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe containing the data.
    x_col : str
        Column name for the x-axis (e.g., Factor/Parameter).
    y_col : str
        Column name for the y-axis (e.g., ICC values).
    hue_col : str
        Column name for color coding (e.g., Model Name).
    col_col : str
        Column name to define grid columns (e.g., Author/Year).
    row_col : str, optional
        Column name to define grid rows (e.g., Task Name). 
    ref_lines : list of float
        Y-axis positions for horizontal reference lines (e.g., thresholds for reliability).
    palette : str or list
        Color palette for the stripplot.
    height : float
        Height (in inches) of each facet.
    aspect : float
        Aspect ratio of each facet, so that width = aspect * height.
    sharey : bool
        If True, facets share the same Y-axis limits.

    Returns:
    --------
    g : sns.FacetGrid
        The seaborn FacetGrid object.
    """
    
    # 1. Data Preparation
    # Ensure we only work with rows that have data for the plotting dimensions
    cols_to_check = [x_col, hue_col, col_col]
    if row_col:
        cols_to_check.append(row_col)
    
    plot_data = df.dropna(subset=cols_to_check).copy()

    plot_data[x_col] = plot_data[x_col].str.replace("_", " ") 
    
    # Handle FacetGrid arguments: col_wrap cannot be used if row is specified
    grid_args = {
        'col': col_col,
        'height': height,
        'aspect': aspect,
        'sharex': True, # Keep X shared for alignment
        'sharey': sharey
    }
    
    if row_col:
        grid_args['row'] = row_col
        grid_args['margin_titles'] = True # Better labeling for rows
    else:
        grid_args['col_wrap'] = 3

    g = sns.FacetGrid(plot_data, **grid_args)
    
    # 3. Define Custom Plotting Function
    def draw_box_and_strip(data, **kws):
        ax = plt.gca()
        
        # Layer A: Aggregate Boxplot (Gray/Neutral)
        sns.boxplot(
            data=data, 
            x=x_col, 
            y=y_col,
            color='white',       # Neutral background
            showfliers=False,    # Hide outliers (stripplot will show them)
            width=0.5,
            linewidth=1.2,
            ax=ax,
            boxprops={'edgecolor': 'gray', 'alpha': 0.8},
            zorder=1
        )
        
        # Layer B: Colored Stripplot (By Hue)
        sns.stripplot(
            data=data, 
            x=x_col, 
            y=y_col,
            hue=hue_col,         # Color mapping happens here
            palette=palette,   
            dodge=False,         # Keep points centered on the factor
            jitter=0.2,
            size=6,
            alpha=0.8,
            edgecolor='white',
            linewidth=0.5,
            ax=ax,
            zorder=2,
            legend=False         # We manually build the legend later
        )

    # 4. Apply Function to Grid
    g.map_dataframe(draw_box_and_strip)
    
    # 6. Manage Legend
    # Extract handles/labels from a dummy plot to ensure correct mapping
    # (Since map_dataframe doesn't always bubble up the legend handles easily)
    if hue_col and not plot_data.empty:
        # Create a dummy stripplot just to get legend handles
        dummy_ax = plt.figure().add_subplot(111)
        sns.stripplot(data=plot_data, x=x_col, y=y_col, hue=hue_col, palette=palette, ax=dummy_ax)
        handles, labels = dummy_ax.get_legend_handles_labels()
        plt.close(dummy_ax.figure) # Close dummy figure

        # Place Legend outside the grid
        g.fig.legend(
            handles, 
            labels, 
            loc='center right', 
            bbox_to_anchor=(1.2, 0.5), 
            title="",
            frameon=False
        )

    # 5. Add Reference Lines and Formatting
    for ax in g.axes.flat:
        # Draw reference lines from the parameter list
        colors = ['red', 'green', 'orange']
        for i, ref in enumerate(ref_lines):
            c = colors[i] if i < len(colors) else 'gray'
            ax.axhline(ref, color=c, linestyle='--', alpha=0.9, linewidth=1, zorder=0)
        
        # Rotate X-axis labels for readability
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
            label.set_fontsize(10)

    
    # 7. Final Polish
    g.set_axis_labels("", "ICC")
    g.set_titles(row_template="{row_name}", col_template="{col_name}",fontweight='bold')
    
    return g



def plot_distance_comparison(
    distance_df, 
    ax=None, 
    figsize=(6, 5), 
    save_path=None, 
    fontsize_label=15, 
    fontsize_tick=14, 
    palette='viridis',
    stat_test='mannwhitneyu',
    **kwargs
):
    """
    Visualize the distribution of distances between tasks vs authors with statistical annotation.

    Parameters
    ----------
    distance_df : pd.DataFrame
        Output of compute_centroid_distances combined for both groupings
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    figsize : tuple, default (6, 5)
        Figure size if ax is None.
    save_path : str, optional
        Path to save the figure to.
    fontsize_label : int, default 15
        Font size for axis labels.
    fontsize_tick : int, default 14
        Font size for axis ticks.
    palette : str, default 'viridis'
        Color palette for seaborn.
    stat_test : str, default 'mannwhitneyu'
        Statistical test to use: 'mannwhitneyu' or 'ttest_ind'.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats

    # Handle alias for backwards compatibility
    fontsize_label = kwargs.get('label_fontsize', fontsize_label)
    fontsize_tick = kwargs.get('tick_fontsize', fontsize_tick)

    # Set Seaborn theme
    sns.set_theme(style="whitegrid", rc={"axes.grid": True})

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Data processing
    df = distance_df.copy()
    label_map = {'task_name': 'Across Tasks', 'author_year': 'Across Labs'}
    
    group_col = 'Grouping'
    if group_col in df.columns:
        if set(label_map.keys()).issubset(df[group_col].unique()):
            df[group_col] = df[group_col].map(label_map)
        
        # Plotting: Boxplot + Stripplot
        sns.boxplot(x=group_col, y='Distance', data=df, 
                    palette=palette, width=0.5, ax=ax, linewidth=1.3,
                    boxprops=dict(alpha=0.8)) 
        
        sns.stripplot(x=group_col, y='Distance', data=df, 
                      color='.25', size=4, alpha=0.6, jitter=True, ax=ax)

        # Statistical Testing
        groups = df[group_col].unique()
        if len(groups) == 2:
            group1_data = df[df[group_col] == groups[0]]['Distance']
            group2_data = df[df[group_col] == groups[1]]['Distance']
            
            # Calculate p-value
            if stat_test == 'mannwhitneyu':
                stat, p_val = stats.mannwhitneyu(group1_data, group2_data)
            else:
                stat, p_val = stats.ttest_ind(group1_data, group2_data)
            
            # Determine significance stars
            if p_val > 0.05: text = 'ns'
            elif p_val > 0.01: text = '*'
            elif p_val > 0.001: text = '**'
            else: text = '***'
            
            # Draw bracket and text
            y_max = df['Distance'].max()
            y_min = df['Distance'].min()
            y_range = y_max - y_min
            
            y_h = y_max + y_range * 0.05 
            y_text = y_h + y_range * 0.02
            
            x1, x2 = 0, 1
            line_height = y_range * 0.02
            ax.plot([x1, x1, x2, x2], [y_h, y_h + line_height, y_h + line_height, y_h], lw=1.4, c='k')
            
            # Format p-value string
            p_text_val = f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.3f}"
            ax.text((x1 + x2) * 0.5, y_text, f"{text}\n(p={p_text_val})", 
                    ha='center', va='bottom', color='k', fontsize=fontsize_tick)
            
            # Adjust y-limit
            ax.set_ylim(top=y_text + y_range * 0.15)

    # Formatting axes
    ax.set_ylabel('Euclidean Distance', fontsize=fontsize_label, fontweight='bold')
    ax.set_xlabel('', fontsize=0) 
    
    ax.tick_params(axis='x', labelsize=fontsize_tick)
    ax.tick_params(axis='y', labelsize=fontsize_tick)
    
    # Remove top and right spines
    sns.despine(trim=True, ax=ax)

    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight')
    
    return ax

