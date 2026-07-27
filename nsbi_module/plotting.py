import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Union
from scipy.stats import pearsonr,f_oneway, ttest_ind
import statsmodels.api as sm
import matplotlib.lines as mlines
from typing import Dict, Any, List, Optional, Tuple

def plot_rt_dists(data, flip=True, required_columns = ["rt", "accuracy", "subject_id"], **kwargs):

    assert all(
        col in data.columns for col in required_columns
    ), f"DataFrame is missing one or more of the required columns: {required_columns}"

    if flip:
        data[required_columns[0]] = np.where(data[required_columns[1]] == 1, data[required_columns[0]], -data[required_columns[0]])
        hue = "congruency"
    else:
        hue = required_columns[1]

    g = sns.FacetGrid(
        data, col=required_columns[2], sharex=False, sharey=False, hue=hue, col_wrap=5
    )
    g.map(sns.histplot, required_columns[0], kde=True, **kwargs)

    g.set_titles(col_template="subj_({col_name})")
    g.set_xlabels("Response Time (second)")
    g.set_ylabels("Frequency")

    plt.legend()
    plt.show()

def plot_ppc_rt_dists(
    obs_data,
    ppc_data,
    col="subject_id",
    hue="congruency",
    dv="rt",
    alpha=0.4,
    palette=None
):

    g = sns.FacetGrid(
        obs_data,
        col=col,
        sharex=False,
        sharey=False,
        hue=hue,
        col_wrap=5,
        palette=palette
    )

    g.map(sns.histplot, dv, kde=False, alpha=alpha, linewidth=0, stat='density')
    g.map(sns.kdeplot, dv, data=ppc_data, fill=False, common_norm=False)

    g.set_titles(col_template="subj_({col_name})")
    g.set_xlabels("Response Time (second)")
    g.set_ylabels("Frequency")

    plt.legend()
    plt.show()


def plot_params_recovery_by_nobs(
    df: pd.DataFrame,
    ground_truth: Union[pd.DataFrame, np.ndarray, pd.Series] = None,
    x="n_obs",
    y="mean",
    n_col=4,
    ticks_size=12,
    ticks_rotation=55,
    y_label_size=18,
    facet_title_size=18,
    log_x=False,
    x_values=np.array([50, 100, 200, 500, 1000, 2000, 5000]),
    **kwargs
):

    assert "params" in df.columns, "df must have column 'params'"

    x_title = kwargs.pop("x_title", 'Number of Observations')
    y_title = kwargs.pop("y_title", 'Mean Estimates')

    g = sns.FacetGrid(df, col="params", sharey=False, col_wrap=n_col)
    g.map_dataframe(sns.regplot, x=x, y=y, **kwargs)
    g.set_titles(col_template="{col_name}", size=facet_title_size)

    # g.set_axis_labels('Number of Observations', 'Mean Estimates')
    g.set_axis_labels('', '')
    g.figure.text(0.5, 0, x_title, ha='center', va='center', fontsize=ticks_size + 2)
    g.figure.text(
        0,
        0.5,
        y_title,
        ha='center',
        va='center',
        rotation='vertical',
        fontsize=ticks_size + 2
    )

    if log_x:
        g.set(xscale="log")

        # x_values = np.array(df[x].unique())

        def format_xaxis(ax):
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: '{:.0f}'.format(y))
            )
            ax.set_xticks(x_values)
            ax.set_xticklabels([f'{v:.0f}' for v in x_values])
            ax.xaxis.set_tick_params(rotation=ticks_rotation, labelsize=ticks_size)
            ax.yaxis.set_tick_params(labelsize=ticks_size)
            ax.set_ylabel(ax.get_ylabel(), fontsize=y_label_size)

        for ax in g.axes.flat:
            format_xaxis(ax)

    if ground_truth is not None:
        for idx, ax in enumerate(g.axes.flat):
            title = ax.get_title()
            if isinstance(ground_truth, pd.DataFrame):
                ref_value = ground_truth[title][0]
            if isinstance(ground_truth, pd.Series):
                ref_value = ground_truth[title]
            elif isinstance(ground_truth, np.ndarray):
                ref_value = ground_truth[idx]
            ax.axhline(y=ref_value, color='r', linestyle='--')

    plt.tight_layout()
    plt.show()


def regplot_with_corr(
    data=None,
    x="x",
    y="y",
    cor_anonot=True,
    reg_anonot=True,
    annot_kws={
        "fontsize": 8,
        "xy": (0.95, 0.05),
        "ha": 'right',
        "va": 'bottom'
    },
    scatter_kws={
        's': 40,
        "alpha": 0.4
    },
    ax=None,
    **kwargs
):
    """

    Example:
    --------
    >>> Example usage
    >>> import pandas as pd
    >>> data = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [2, 3, 5, 7, 11]})
    >>> regplot_with_corr(data)
    >>> plt.show()
    """
    if ax is None:
        ax = plt.gca()
    if data is not None:
        data_x = data[x]
        data_y = data[y]

    # Plot regression line and scatter plot
    sns.regplot(
        x=data_x,
        y=data_y,
        ci=None if len(np.unique(data_y)) == 1 else 95,
        scatter_kws=scatter_kws,
        ax=ax
    )

    # Annotate the plot with correlation, p-value (significance), intercept, and slope
    annot_text = ""
    if cor_anonot:
        # Calculate Pearson correlation
        correlation, p_value = pearsonr(data_x, data_y)
        # if np.isnan(correlation):
        #     correlation = 0
        # if np.isnan(p_value):
        #     p_value = 1
        p_str = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
        annot_text += f"$r={correlation:.2f}$\n${p_str}$"

    if reg_anonot:
        # Calculate regression coefficients
        X = sm.add_constant(data_x)  # Adds a constant term to the predictor
        model = sm.OLS(data_y, X).fit()
        intercept, slope = model.params
        annot_text += f"\n$\\beta_0={intercept:.2f}$\n$\\beta_1={slope:.2f}$"

    # Annotate the plot with correlation, p-value, intercept, and slope
    if annot_text != "":
        ax.annotate(
            annot_text,
            **annot_kws,
            xycoords='axes fraction',
            bbox=dict(
                boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'
            )
        )


def plot_param_cor(df, x='$tau$', xlabel=None, n_col=5, cor_label=True):

    df_long = pd.melt(df, id_vars=[x], var_name='params', value_name='value')
    g = sns.FacetGrid(
        df_long, col="params", col_wrap=n_col, sharex=True, sharey=False
    )

    g.map_dataframe(sns.regplot, x, "value")

    def annotate(data, **kws):
        r, p = pearsonr(data[x], data['value'])
        p_str = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        ax = plt.gca()
        ax.annotate(
            f'r = {r:.2f}\n{p_str}',
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            ha='left',
            va='top',
            fontsize=9,
            bbox=dict(
                boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'
            )
        )

    g.map_dataframe(annotate)

    g.set_axis_labels(x_var=x if xlabel is None else xlabel, y_var="")
    g.set_titles(col_template="{col_name}")

    plt.tight_layout()
    plt.show()


def boxplot_with_significance(data, x='identity', dv='value', hue=None, **kwargs):
    
    show_anova= kwargs.pop("anova", True)
    show_ttest = kwargs.pop("ttest", True)
    
    if hue is None: 
        hue = x

    ax = sns.boxplot(
        x=x,
        y=dv,
        hue=hue,
        legend=False,
        data=data,
        palette="Blues",
        width=0.5,
        showfliers=False,
        **kwargs
    )
    sns.swarmplot(
        x=x,
        y=dv,
        data=data,
        color='gray',
        alpha=0.4,
        size=kwargs.pop("point_size", 5),
        ax=ax
    )

    # ANOVA
    if show_anova:
        anova_result = f_oneway(
            data[data[x] == 'self'][dv],
            data[data[x] == 'friend'][dv],
            data[data[x] == 'other'][dv]
        )
        p_str = "p < 0.001" if anova_result.pvalue < 0.001 else f"p = {anova_result.pvalue:.3f}"
        ax.annotate(
            f'ANOVA $F={anova_result.statistic:.3f}$ \n ${p_str}$',
            xy=(0.35, 0.1),
            xycoords='axes fraction',
            ha='left',
            size=8
        )

    # T-test
    if show_ttest:
        # param = data['variable'].unique()[0]
        pairs = list(itertools.combinations(data[x].unique(), 2))
        y_max = data[dv].max()
        y_min = data[dv].min()
        y_range = y_max - y_min
        for i, (group1, group2) in enumerate(pairs):
            data1 = data[data[x] == group1][dv]
            data2 = data[data[x] == group2][dv]
            t_stat, p_val = ttest_ind(data1, data2)

            if p_val < 0.05:
                x1, x2 = data[x].unique().tolist(
                ).index(group1), data[x].unique().tolist().index(group2)
                y, h, col = y_max + y_range * 0.05 * (i + 1), y_range * 0.02, 'k'
                ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
                ax.text((x1 + x2) * .5, y + h, "*", ha='center', va='bottom', color=col)

def plot_single_radar(
    data=None, 
    variables=None, 
    hue=None, 
    show_ticks=False, 
    show_legend=True,
    draw_fill=True,
    title=None, 
    ax=None, 
    **kwargs
    ):
    """
    Helper function to plot radar charts with multiple groups, customizable circle layers, and value ticks.
    
    Parameters:
    - data: pandas DataFrame containing the data.
    - ax: matplotlib Axes object to plot the radar chart.
    - variables: List of variable names (columns) to plot.
    - hue: Column name for grouping data with different colors.
    - title: Title of the radar plot.
    - show_ticks: Boolean indicating whether to show the tick values on the circles.
    """
    
    assert data is not None, "Please provide a pandas DataFrame for the data."
    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4), dpi=120, subplot_kw=dict(polar=True))

    # Determine number of variables and angles
    num_vars = len(variables)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the radar chart
    
    unique_hues = data[hue].unique() if hue else [None]
    for hue_val in unique_hues:
        group_data = data[data[hue] == hue_val] if hue else data
        values = group_data[variables].mean().tolist()
        values += values[:1]  # Add the first value to the end to close the radar chart
        
        if draw_fill:
            ax.fill(angles, values, alpha=0.3)  # Fill the area for each group
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=hue_val)  # Plot the outline for each group

    # Set labels and ticks for the radar chart
    if not show_ticks:
        ax.set_yticklabels([])  # Hide the y-axis labels
    ax.set_xticks(angles[:-1])  # Set the angles for the x-axis
    ax.set_xticklabels(variables)  # Set the variable names as the x-axis labels
    
    # Set the title for the plot (if title is provided)
    if title:
        ax.set_title(title, size=16)

    # Show legend if there are multiple groups
    if hue is not None and show_legend:
        ax.legend(loc='best')

def plot_radar(
    data=None, variables=None, 
    facet_col=None, 
    hue=None, 
    title=None, 
    height = 4, aspect = 1,
    show_ticks=False, show_legend=True,
    draw_fill=False,
    legend_kws = dict(loc='lower center',ncols=5),
    **kwargs):
    """
    Create radar charts for each facet in the DataFrame.
    
    Parameters:
    - data: pandas DataFrame containing the data.
    - variables: List of variable names (columns) to plot.
    - hue: Column name for grouping data with different colors.
    - facet_col: Column name to facet the data by (create multiple plots based on this column).
    - title: Title of the radar plot.
    - show_ticks: Boolean indicating whether to show the tick values on the circles.
    - show_legend: Boolean indicating whether to show the legend.
    """
    
    # Create the FacetGrid if facet_col is provided
    g = sns.FacetGrid(
        data, 
        col=facet_col, height=height, aspect=aspect, 
        subplot_kws = dict(polar=True),
        despine = False,
        **kwargs)
    
    _plot_radar = plot_single_radar
    _plot_radar.__module__ = "seaborn_polar_radar"
    
    # Map the plotting function to each facet
    g.map_dataframe(_plot_radar, variables=variables, hue=hue, show_ticks=show_ticks, show_legend=show_legend,draw_fill=draw_fill)

    # Adjust layout to avoid overlap
    g.set_titles("{col_name}")

    if hue is not None:
        g.add_legend(**legend_kws)

    if title is not None:
        g.figure.suptitle(title, y=1.05)

    return g

def plot_goodness_of_fit(
    plotting_data: Dict[str, Any],
    metric_type: str = "rt",
    subject_col: str = "Subject",
    condition_col: str = "Comp",
    rt_col: str = "rt_cor",
    err_col: str = "per_err",
    colors: Dict[str, str] = {"comp": "#1f77b4", "incomp": "#d62728"},  # Blue vs Red
    subplot_size: int = 3,
    scatter_size: int = 25,
    scatter_alpha: int = 0.6,
    save_name: Optional[str] = None
):
    """
    Generates a grid of scatter plots comparing Observed vs. Predicted means per subject.
    
    Improvements:
    - 1:1 Aspect Ratio for perfect square plots.
    - Dynamic figure size based on N models and N datasets.
    - Simplified titles.
    - SVG saving support.
    
    Args:
        plotting_data (Dict): Structured data dictionary.
        metric_type (str): 'rt' or 'acc'. Defaults to "rt".
        subject_col (str): Subject ID column.
        condition_col (str): Condition column (Comp/Incomp).
        rt_col (str): RT column name.
        err_col (str): Error % column name.
        colors (Dict): Colors for Congruent/Incongruent points.
        save_name (str, optional): If provided, saves figure to this path (e.g., 'plot.svg').
    """
    
    # 1. Determine Grid Dimensions
    # -------------------------------------------------------
    datasets = list(plotting_data.keys())
    if not datasets: return
    models = list(plotting_data[datasets[0]]["models"].keys())
    
    n_rows, n_cols = len(datasets), len(models)
    
    # Dynamic Figure Size: ~4 inches per subplot
    figsize = (n_cols * subplot_size, n_rows * subplot_size)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    
    # Determine columns and labels based on metric type
    if metric_type == "rt":
        val_col = rt_col
        label_str = "Mean RT (ms)"
    else:
        val_col = "calculated_acc"
        label_str = "Accuracy (%)"

    # 2. Plotting Loop
    # -------------------------------------------------------
    for i, ds_name in enumerate(datasets):
        obs_means = plotting_data[ds_name]["observed"]["subject_means"].copy()
        
        # Pre-process Accuracy (Error % -> Accuracy %)
        if metric_type == "acc":
            obs_means[val_col] = 100 - obs_means[err_col]

        for j, model_name in enumerate(models):
            ax = axes[i, j]
            pred_means = plotting_data[ds_name]["models"][model_name]["subject_means"].copy()
            
            if metric_type == "acc":
                pred_means[val_col] = 100 - pred_means[err_col]

            # Merge Data
            merged = pd.merge(
                obs_means, 
                pred_means, 
                on=[subject_col, condition_col], 
                suffixes=('_obs', '_pred')
            )
            
            # Determine Axis Limits (Crucial for 1:1 Visuals)
            # Find global min/max for this specific subplot to ensure squareness
            all_vals = pd.concat([merged[f"{val_col}_obs"], merged[f"{val_col}_pred"]])
            min_val = all_vals.min()
            max_val = all_vals.max()
            
            # Add some padding (5%)
            padding = (max_val - min_val) * 0.05
            limit_min, limit_max = min_val - padding, max_val + padding
            
            # Draw Diagonal (Perfect Fit)
            ax.plot([limit_min, limit_max], [limit_min, limit_max], 'k--', alpha=0.3, zorder=0)

            # Scatter Plot by Condition
            unique_conds = merged[condition_col].unique()
            
            for cond in unique_conds:
                subset = merged[merged[condition_col] == cond]
                
                # Determine color based on condition name
                c_key = cond
                color = colors.get(c_key, "gray")
                
                ax.scatter(
                    subset[f"{val_col}_obs"], 
                    subset[f"{val_col}_pred"], 
                    c=color, 
                    alpha=scatter_alpha, 
                    edgecolor='white',
                    s=scatter_size, 
                    zorder=10
                )

            # Correlation Text
            r_val, _ = pearsonr(merged[f"{val_col}_obs"], merged[f"{val_col}_pred"])
            ax.text(0.05, 0.9, f"$r = {r_val:.2f}$", transform=ax.transAxes, fontsize=12, fontweight='bold')

            # 3. Styling & Constraints
            # -------------------------------------------------------
            # Aspect Ratio 1:1
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(limit_min, limit_max)
            ax.set_ylim(limit_min, limit_max)
            
            # Titles (Simplified: just "DDM", not "Model: DDM")
            if i == 0:
                ax.set_title(model_name, fontsize=14, fontweight='bold', pad=15)
            
            # Row Labels (Dataset Name)
            if j == 0:
                ax.set_ylabel(f"Predicted {label_str}", fontsize=11)
                # Place Dataset Name to the left of the Y-axis
                ax.text(-0.4, 0.5, ds_name, transform=ax.transAxes, 
                        rotation=90, va='center', ha='right', fontweight='bold', fontsize=12)
            else:
                ax.set_ylabel("") # Hide Y labels for inner plots
                
            # X Labels
            if i == n_rows - 1:
                ax.set_xlabel(f"Observed {label_str}", fontsize=11)
            else:
                ax.set_xlabel("") # Hide X labels for inner plots

    # 4. Custom Legend (Fixed Colors)
    # -------------------------------------------------------
    # We manually create handles to ensure the legend is perfectly clean
    legend_handles = [
        mlines.Line2D([], [], color='w', marker='o', markerfacecolor=colors['comp'], 
                      markersize=10, label='Congruent'),
        mlines.Line2D([], [], color='w', marker='o', markerfacecolor=colors['incomp'], 
                      markersize=10, label='Incongruent')
    ]
    
    fig.legend(handles=legend_handles, loc='lower center', ncol=2, 
               bbox_to_anchor=(0.5, 0.0), fontsize=12, frameon=False)

    # Layout Adjustments
    # Use tight_layout but leave space at bottom for legend and left for row labels
    plt.tight_layout() 
    
    # 5. Saving
    # -------------------------------------------------------
    if save_name:
        plt.savefig(save_name, format="svg", bbox_inches='tight')
        print(f"Figure saved to {save_name}")
        
    return fig


def _get_grid_dims(plotting_data: Dict) -> Tuple[List[str], List[str]]:
    """Helper to extract dataset names (rows) and model names (cols) from data."""
    datasets = list(plotting_data.keys())
    # Assume all datasets have the same models for simplicity, taking from the first one
    if not datasets:
        return [], []
    models = list(plotting_data[datasets[0]]["models"].keys())
    return datasets, models

# ─── Distribution plotting function (CAF/CDF) ─────────────────────────────────
def plot_distribution_curve(
    plotting_data: Dict[str, Any],
    plot_type: str = "caf",
    x_col: str = "bin",
    y_cols: Dict[str, str] = {"comp": "comp", "incomp": "incomp"},
    line_styles: Dict[str, str] = {"comp": "-", "incomp": "--"},
    model_colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[float, float] = (3.5, 2.5),
    save_name: Optional[str] = None,
    known_tasks: List[str] = None,
    task_order: List[str] = None,
    study_label_fontsize: int = 9,
    study_label_fontweight: str = "normal",
    tick_labelsize: int = 8,
) -> Optional[plt.Figure]:
    """
    Generates a grid of CAF or CDF curves with a structured layout.
    
    Layout:
      - Rows: Studies
      - Columns: Tasks
      - Cells: All models and observed data are plotted together in one axis.
    
    Visual Mapping:
      - Observed Data : Black
      - Model Data    : Defined by `model_colors` (e.g., 4 different colors)
      - Condition     : Solid line for Congruent ('comp'), Dashed for Incongruent ('incomp')
    """
    datasets, models = _get_grid_dims(plotting_data)
    if len(datasets) == 0:
        return None

    # Default colors if none provided
    if model_colors is None:
        default_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        model_colors = {m: default_palette[i % len(default_palette)] for i, m in enumerate(models)}

    # Restructure data into a nested dict: structured[study][task]
    structured, studies, tasks_col = restructure_plotting_data(
        plotting_data, known_tasks, task_order
    )
    n_rows, n_cols = len(studies), len(tasks_col)

    # Create figure
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize[0] * n_cols, figsize[1] * n_rows),
        squeeze=False
    )

    models_in_legend = set()

    for i, study in enumerate(studies):
        # Capitalize first letter of study name for display
        study_label = study[0].upper() + study[1:]

        for j, task in enumerate(tasks_col):
            ax = axes[i, j]

            # 1. Formatting: Titles and Labels
            if i == 0:
                ax.set_title(task, fontsize=11, fontweight='bold', pad=6)

            if j == 0:
                y_label_text = "Accuracy" if plot_type == 'caf' else "RT (ms)"
                ax.set_ylabel(
                    f"{study_label}\n\n{y_label_text}",
                    fontsize=study_label_fontsize,
                    fontweight=study_label_fontweight,
                    labelpad=4,
                )

            if i == n_rows - 1:
                ax.set_xlabel("RT Bin / Quantile", fontsize=9)

            # 2. Handle missing data cells safely (preserving outer labels)
            if task not in structured[study]:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                continue

            data = structured[study][task]
            
            # Grid styling
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)

            # 3. Plot Observed Data (Black)
            if "observed" in data and plot_type in data["observed"]:
                obs_df = data["observed"][plot_type]
                for cond_key, col_name in y_cols.items():
                    style = line_styles.get(cond_key, "-")
                    if col_name in obs_df.columns:
                        ax.plot(
                            obs_df[x_col], 
                            obs_df[col_name], 
                            color="black", 
                            linestyle=style, 
                            marker="o", 
                            markersize=4,
                            linewidth=2
                        )

            # 4. Plot Model Predictions (Colored)
            for model_name in models:
                if "models" not in data or model_name not in data["models"]:
                    continue
                if plot_type not in data["models"][model_name]:
                    continue

                pred_df = data["models"][model_name][plot_type]
                c = model_colors.get(model_name, "#aaaaaa")
                models_in_legend.add(model_name)

                for cond_key, col_name in y_cols.items():
                    style = line_styles.get(cond_key, "-")
                    if col_name in pred_df.columns:
                        ax.plot(
                            pred_df[x_col], 
                            pred_df[col_name], 
                            color=c, 
                            linestyle=style, 
                            linewidth=2
                        )

    # 5. Build Shared Custom Legend
    row1 = [
        mlines.Line2D([], [], color='black', marker='o', markersize=4, linewidth=2, label='Observed'),
        mlines.Line2D([], [], color='gray', linestyle=line_styles.get('comp', '-'), linewidth=2, label='Congruent'),
        mlines.Line2D([], [], color='gray', linestyle=line_styles.get('incomp', '--'), linewidth=2, label='Incongruent'),
        mlines.Line2D([], [], color='none', label='') 
    ]

    row2 = []
    desired_order = ["DDM", "DMC", "SSP", "DSTP"]
    
    for model_name in desired_order:
        if model_name in models_in_legend:
            c = model_colors.get(model_name, "#aaaaaa")
            row2.append(mlines.Line2D([], [], color=c, linewidth=2, label=model_name))
            
    while len(row2) < 4:
        row2.append(mlines.Line2D([], [], color='none', label=''))

    legend_handles = []
    for i in range(4):
        legend_handles.append(row1[i])
        legend_handles.append(row2[i])

    plot_title = "Conditional Accuracy Function (CAF)" if plot_type == 'caf' else "Cumulative Distribution Function (CDF)"
    fig.suptitle(plot_title, fontsize=12, fontweight='bold')

    fig.legend(
        handles=legend_handles, 
        loc='lower center', 
        ncol=4,  
        bbox_to_anchor=(0.5, 0.0),
        fontsize=9,
        frameon=True,
        fancybox=True,
        shadow=True
    )
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.96]) 

    # 6. Saving
    if save_name:
        plt.savefig(save_name, format="svg", bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_name}")
        
    return fig


# ─── Helper: split flat dataset key into study + task ─────────────────────────
def parse_dataset_name(
    name: str,
    known_tasks: List[str] = None,
) -> Tuple[str, str]:
    """
    Split a flat dataset key such as 'clayson2025flanker' into
    a study identifier and a task label.

    Parameters
    ----------
    name        : flat dataset key (e.g. 'eisenberg2019flanker')
    known_tasks : lower-case task suffixes to detect

    Returns
    -------
    (study, task)  – task is title-cased; returns (name, 'Unknown') if not matched.

    Examples
    --------
    >>> parse_dataset_name('eisenberg2019flanker')
    ('eisenberg2019', 'Flanker')
    """
    lower = name.lower()
    for task in known_tasks:
        if lower.endswith(task):
            return name[: -len(task)], task.capitalize()
    return name, 'Unknown'


# ─── Helper: restructure flat dict into study × task ─────────────────────────
def restructure_plotting_data(
    plotting_data: Dict[str, Any],
    known_tasks: List[str] = None,
    task_order: List[str] = None,
) -> Tuple[Dict[str, Dict[str, Any]], List[str], List[str]]:
    """
    Convert a flat {dataset_name: data} dict into a nested {study: {task: data}}
    structure suitable for a rows=studies × columns=tasks grid layout.

    Parameters
    ----------
    plotting_data : original flat plotting dict
    known_tasks   : lower-case task suffixes to detect (e.g. ['flanker', 'simon', 'stroop'])
    task_order    : preferred column order, title-cased (e.g. ['Flanker', 'Simon', 'Stroop'])

    Returns
    -------
    structured : {study: {task: data}}
    studies    : alphabetically sorted study identifiers
    tasks      : task names in requested order (only those present in data)

    Examples
    --------
    >>> structured, studies, tasks = restructure_plotting_data(final_data)
    >>> studies[:3]
    ['clayson2025', 'eisenberg2019', 'hedge2018']
    >>> tasks
    ['Flanker', 'Simon', 'Stroop']
    """
    structured: Dict[str, Dict[str, Any]] = {}
    found_tasks: set = set()

    for ds_name, data in plotting_data.items():
        study, task = parse_dataset_name(ds_name, known_tasks)
        structured.setdefault(study, {})[task] = data
        found_tasks.add(task)

    studies = sorted(structured.keys())

    # Preserve task_order; append any unrecognized tasks alphabetically at the end
    ordered_tasks = [t for t in task_order if t in found_tasks]
    ordered_tasks += sorted(found_tasks - set(task_order))

    return structured, studies, ordered_tasks


# ─── Main plotting function ───────────────────────────────────────────────────
def plot_delta_functions(
    plotting_data: Dict[str, Any],
    x_col: str = "mean_bin",
    y_col: str = "mean_effect",
    model_colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[float, float] = (3.5, 2.5),
    save_name: Optional[str] = None,
    structured_layout: bool = False,
    known_tasks: List[str] = None,
    task_order: List[str] = None,
    max_cols: int = 5,
    auto_layout: bool = True,
    study_label_fontsize: int = 9,
    study_label_fontweight: str = "normal",
    tick_labelsize: int = 8,
) -> Optional[plt.Figure]:
    """
    Plot delta functions showing the conflict effect over time.

    Layout modes
    ------------
    structured_layout=False  (default)
        Flat grid: one panel per dataset, auto-wrapped into rows.
    structured_layout=True
        Organized grid: rows = study identifiers, columns = task types.
        Dataset keys are parsed with parse_dataset_name() to split
        e.g. 'eisenberg2019flanker' → study='eisenberg2019', task='Flanker'.

    Parameters
    ----------
    plotting_data     : nested dict {dataset_name: {"observed": ..., "models": ...}}
    x_col             : column name for the mean-RT bin
    y_col             : column name for the delta effect
    model_colors      : dict mapping model names to hex colors; defaults to MODEL_COLORS
    figsize           : (width, height) of a single subplot panel in inches
    save_name         : file path to save as SVG (optional)
    structured_layout : if True, use rows=studies × columns=tasks grid
    known_tasks       : lower-case task suffixes to detect (structured_layout only)
    task_order        : preferred column order, title-cased (structured_layout only)
    max_cols          : max columns in flat auto-layout mode
    auto_layout       : whether to wrap into multiple rows in flat mode

    Returns
    -------
    fig : matplotlib Figure object
    """

    datasets, models = _get_grid_dims(plotting_data)
    n_datasets = len(datasets)
    if n_datasets == 0:
        return None

    # ── Determine grid dimensions ─────────────────────────────────────────
    if structured_layout:
        structured, studies, tasks_col = restructure_plotting_data(
            plotting_data, known_tasks, task_order
        )
        n_rows, n_cols = len(studies), len(tasks_col)
    else:
        if auto_layout and n_datasets > max_cols:
            n_cols = min(max_cols, n_datasets)
        else:
            n_cols = n_datasets
        n_rows = int(np.ceil(n_datasets / n_cols))

    # ── Create figure ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize[0] * n_cols, figsize[1] * n_rows),
        squeeze=False,
    )

    # ── Build shared legend handles ───────────────────────────────────────
    legend_handles = [
        mlines.Line2D(
            [], [], color='black', linestyle='-',
            linewidth=2, marker='o', markersize=4, label='Observed'
        )
    ]
    models_in_legend: set = set()

    # ── Inner helper: draw observed + model curves into one axis ──────────
    def _plot_cell(ax: plt.Axes, data: Dict[str, Any]) -> None:
        """Render observed data and all model predictions onto a single axis."""
        # Observed data: black solid line with markers
        obs_df = data["observed"]["delta"]
        ax.plot(
            obs_df[x_col], obs_df[y_col],
            color='black', linestyle='-', linewidth=2, marker='o', markersize=4
        )

        # Model predictions: colored dashed lines
        for model_name in models:
            if model_name not in data.get("models", {}):
                continue
            c = model_colors.get(model_name, '#aaaaaa')
            pred_df = data["models"][model_name]["delta"]
            ax.plot(
                pred_df[x_col], pred_df[y_col],
                color=c, linestyle='--', linewidth=2
            )
            if model_name not in models_in_legend:
                legend_handles.append(
                    mlines.Line2D(
                        [], [], color=c, linestyle='--',
                        linewidth=2, label=model_name
                    )
                )
                models_in_legend.add(model_name)

        # Formatting
        ax.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
        ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # ── Structured layout: rows = studies, columns = tasks ────────────────
    if structured_layout:
        for i, study in enumerate(studies):
            # Capitalize first letter of study name for display
            study_label = study[0].upper() + study[1:]

            for j, task in enumerate(tasks_col):
                ax = axes[i, j]

                # 👇 修改点 1：将标题和标签的绘制逻辑，移动到 continue 跳出检查之前
                # Column header: show task name only on the top row
                if i == 0:
                    ax.set_title(task, fontsize=11, fontweight='bold', pad=6)

                # Row label: show study identifier on the leftmost column only
                if j == 0:
                    ax.set_ylabel(
                        f"{study_label}\n\nDelta Effect\n(Incomp − Comp)",
                        fontsize=study_label_fontsize,
                        fontweight=study_label_fontweight,
                        labelpad=4,
                    )

                # X-axis label on the bottom row only
                if i == n_rows - 1:
                    ax.set_xlabel("Mean RT (ms)", fontsize=9)

                # Empty cell: this study has no data for this task
                if task not in structured[study]:
                    # 👇 修改点 2：避免使用 ax.axis('off')，否则会吞掉边缘空白图表的 x/y label。
                    # 改为仅隐藏刻度和边框线，保留 title 和 labels。
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    continue

                # 正常绘制数据
                _plot_cell(ax, structured[study][task])
    
    # ── Flat layout: one panel per dataset ───────────────────────────────
    else:
        axes_flat = axes.flatten()
        for idx, ds_name in enumerate(datasets):
            ax = axes_flat[idx]
            _plot_cell(ax, plotting_data[ds_name])

            ax.set_title(ds_name, fontsize=10)
            if idx % n_cols == 0:
                ax.set_ylabel("Delta Effect\n(Incomp − Comp)", fontsize=9)
            if idx >= (n_rows - 1) * n_cols:
                ax.set_xlabel("Mean RT (ms)", fontsize=9)

        # Hide unused panels in the last row
        for idx in range(n_datasets, len(axes_flat)):
            axes_flat[idx].axis('off')

    # ── Global decorations ────────────────────────────────────────────────
    fig.suptitle(
        'Delta Plot: Conflict Effect over Time',
        fontsize=12, fontweight='bold'
    )
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=min(len(legend_handles), 6),
        bbox_to_anchor=(0.5, 0.0),
        fontsize=9,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])

    if save_name:
        plt.savefig(save_name, format="svg", bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_name}")

    return fig


