#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pingouin as pg
from sklearn.preprocessing import StandardScaler

import sys
from nsbi_module.utils_ind_diff import *

ipython = globals().get("get_ipython")
if ipython:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")


# In[2]:


MODELS = ['DDM', 'SSP', 'DMC', 'DSTP']
COLORS = ['#b0d97c', '#92d28e', '#95d8c3', '#81cef0']
MODEL_COLORS = dict(zip(MODELS, COLORS))
COLORS_DARKER = ['#9dd25b', '#6fc96f', '#75cfb6', '#4dc3eb']
MODEL_COLORS_DARKER = dict(zip(MODELS, COLORS_DARKER))


# In[3]:


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
    ax.axhline(0.4, linestyle="--", color="gray", alpha=0.5)

    # Add legend if grouped
    if group is not None and group in df.columns:
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=color_map[grp]) for grp in unique_groups
        ]
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
    if param_str.startswith("$") and param_str.endswith("$"):
        return param_str

    # Handle underscore formatting
    if "_" in param_str:
        # Split by underscore and wrap the part after underscore in {}
        parts = param_str.split("_", 1)  # Split only on first underscore
        formatted_param = f"{parts[0]}_{{{parts[1]}}}"
        return f"${formatted_param}$"
    else:
        # No underscore, simple wrapping
        return f"${param_str}$"


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
        group_col_name = "Group"
        unique_groups = ["All"]
        icc_df = icc_df.copy()
        icc_df[group_col_name] = "All"
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
                match = group_data[group_data["parameter"] == comp]

                if not match.empty:
                    icc_val = match["ICC"].mean()
                else:
                    icc_val = None  # Component not found in this group

                # Append row
                rows.append(
                    {
                        group_col_name: group,
                        "Factor": factor,
                        "Component": comp,
                        "ICC": icc_val,
                    }
                )

    result_df = (
        pd.DataFrame(rows)
        .sort_values(
            by=[group_col_name, "Factor", "ICC"], ascending=[True, True, False]
        )
        .reset_index(drop=True)
    )

    return result_df


def process_factor_scores(df, factor_vars):
    """
    Process dataframe to compute standardized factor scores.

    Parameters:
    ----------
    df : pd.DataFrame
        Input dataframe containing parameter values
    factor_vars : dict
        Dictionary mapping factor names to lists of parameter names

    Returns:
    -------
    pd.DataFrame
        Processed dataframe with standardized factor scores
    """

    # Get all unique variables from all factors
    all_vars = sum(factor_vars.values(), [])
    all_vars = list(set(all_vars))

    # Standardize the variables
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    df_std = df.copy()
    df_std[all_vars] = scaler.fit_transform(df[all_vars])

    # Reverse sign for specific variables
    df_std["$r_d|SSP$"] *= -1
    df_std["$v_{incong}|DDM$"] *= -1

    # Compute factor scores as mean of constituent variables
    for factor_name, components in factor_vars.items():
        # Handle the underscore in column name for output
        output_col_name = factor_name.replace("_", " ")
        df_std[output_col_name] = df_std[components].mean(axis=1)

    return df_std


def plot_cognitive_processes_box(
    data,
    x="author_year",
    hue="task_name",
    cognitive_params=[
        "Processing Efficiency",
        "Decision Caution",
        "Non-decision time",
        "Inhibitory process",
    ],
    palette="viridis",
):
    """
    Create box plots for cognitive process parameters, faceted by parameter type.

    Parameters:
    -----------
    data : pandas.DataFrame
        Input data containing the required columns
    x : str, default "author_year"
        Column name for x-axis values
    hue : str, default "task_name"
        Column name for hue grouping
    cognitive_params : list of str, default ["Processing Efficiency", "Automated process", "Caution", "Non decision time"]
        List of cognitive process parameter column names to plot

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plots
    axes : matplotlib.axes.Axes or array of Axes
        The axes objects of the plots
    """
    # Melt the data to long format for plotting
    melted_data = pd.melt(
        data,
        id_vars=[x, hue],
        value_vars=cognitive_params,
        var_name="cognitive_process",
        value_name="value",
    )

    # Set up the FacetGrid with cognitive processes as columns
    g = sns.FacetGrid(
        melted_data,
        row="cognitive_process",
        height=1.8,
        aspect=4,
        sharex=True,
        sharey=False,
    )

    # Map the box plot to each facet
    g.map_dataframe(sns.boxplot, x=x, y="value", hue=hue, palette=palette)

    g.set_axis_labels("")
    g.set_xticklabels(labels=melted_data[x].unique(), rotation=60, ha="right")

    # Remove row titles (facet titles)
    g.set_titles(row_template="")  # Remove row titles

    # Set y-axis label title for all subplots
    for ax, param in zip(g.axes.flat, cognitive_params):
        ax.set_ylabel(param)  # Use the cognitive parameter name as y-axis title

    # Add legend
    g.add_legend(title="", bbox_to_anchor=(1.05, 0.5), loc="center left")

    # Adjust layout to prevent overlap
    g.fig.tight_layout()

    return g.fig, g.axes


# ## Cross task analysis

# In[4]:


subj_indices = pd.read_csv("43_subj_indices_with_EFA_scores.csv")
subj_indices.head()


# In[5]:


four_factors_names = [
    "Processing Efficiency", "Decision Caution", "Non-decision time", "Inhibitory process"
]
df = subj_indices[["subject_id", "task_id", "author_year", "task_name", *four_factors_names]]
fig, axes = plot_cognitive_processes_box(df)


# In[6]:


# it will cost 16 seconds
tmp_subj_indices = subj_indices.query("author_year != 'lee2025'").query("task_id != 'hedge2018simon'")
df_dict = {key_i:tmp_subj_indices.query("author_year == @key_i") for key_i in ['clayson2025', 'eisenberg2019', 'hedge2018', 'whitehead2019', 'ulrich2015', 'reymermet2018']}

factor_vars = {
    "Decision Caution": [
        "$a|DDM$", "$a|SSP$", "$a|DSTP$", "$a|DMC$", 
    ],
    "Non-decision time": [
        "$t|DDM$", "$t|SSP$", "$t|DSTP$", "$t|DMC$"
    ],
    "Processing Efficiency": [
        "$v_{cong}|DDM$", "$p|SSP$", "$v_{c}|DMC$", 
        "$v_{ss}|DSTP$", "$v_{p2}|DSTP$","$v_{ta}|DSTP$", 
    ],
    "Inhibitory process": [
        "$v_{incong}|DDM$", "$sd_a|SSP$", "$r_d|SSP$", "$\\tau|DMC$", "$\\alpha|DMC$", "$\\eta|DMC$", "$v_{fl}|DSTP$"
    ]
}

ignore_cols_1 = ['task_id', 'author_year']
icc_long_format = process_icc_data(
    data=df_dict,                     
    contrast_col='task_name',        
    ignore_cols=ignore_cols_1,
    key_label='author_year',
    factor_vars=factor_vars          
)

# icc_wide = icc_long_format.pivot(index='parameter', columns='author_year', values='ICC')
icc_long_format['author_year'] = icc_long_format['author_year'].apply(format_author_year)
icc_long_format


# In[8]:


sns.set_style("white")
g = plot_icc_distribution(icc_long_format.query("ICC > 0"), palette=MODEL_COLORS_DARKER)
plt.tight_layout()
plt.savefig("../figs/41_parameters_icc_cross_task.svg", format="svg", bbox_inches="tight")


# ## Retest

# In[9]:


subj_indices_retest = pd.read_csv("../03_fitting/23subj_indices_across_models_and_tasks_retest.csv")
subj_indices_retest = subj_indices_retest.query("author_year != 'lee2025'")
subj_indices_retest.head()


# In[10]:


factor_vars = {
    "Control_process": [
        "$v_{cong}|DDM$", "$p|SSP$", "$v_{c}|DMC$", "$v_{incong}|DDM$",
        "$v_{ss}|DSTP$", "$v_{ta}|DSTP$", "$v_{fl}|DSTP$"
    ],
    "Caution": [
        "$a|DDM$", "$a|SSP$", "$a|DSTP$", "$a|DMC$", 
    ],
    "Non_decision_time": [
        "$t|DDM$", "$t|SSP$", "$t|DSTP$", "$t|DMC$"
    ],
    "Automated_process": [
        "$sd_a|SSP$", "$r_d|SSP$", "$v_{fl}|DSTP$", "$v_{incong}|DDM$", "$\\tau|DMC$"
    ]
}

subj_indices_retest = process_factor_scores(subj_indices_retest, factor_vars=factor_vars)


# In[11]:


ignore_cols_2 = ["task_id", "session_id", "author_year", "task_name", "subject_id"]
regex_config = (r'([a-z]+[0-9]{4})([a-z]+)', ['author_year', 'task_name'])

factor_vars = {
    "Caution": [
        "$a|DDM$", "$a|SSP$", "$a|DSTP$", "$a|DMC$", 
    ],
    "Non_decision_time": [
        "$t|DDM$", "$t|SSP$", "$t|DSTP$", "$t|DMC$"
    ],
    "Control_process": [
        "$v_{cong}|DDM$", "$p|SSP$", "$v_{c}|DMC$", 
        "$v_{ss}|DSTP$", "$v_{p2}|DSTP$","$v_{ta}|DSTP$", 
    ],
    "Automated_process": [
        "$v_{incong}|DDM$", "$sd_a|SSP$", "$r_d|SSP$", "$\\tau|DMC$", "$\\alpha|DMC$", "$\\eta|DMC$", "$v_{fl}|DSTP$"
    ]
}

icc_retest_long = process_icc_data(
    data=subj_indices_retest,         
    group_col='task_id',              
    contrast_col='session_id',        
    ignore_cols=ignore_cols_2,
    regex_extract=regex_config,
    factor_vars=factor_vars      
)
icc_retest_long['author_year'] = icc_retest_long['author_year'].apply(format_author_year)
icc_retest_long['task_name'] = icc_retest_long['task_name'].apply(format_task_name)


# In[12]:


g = plot_icc_distribution(icc_retest_long.query("ICC > 0"), palette=MODEL_COLORS_DARKER, row_col="task_name", height=2, aspect=1.1)
plt.tight_layout()
plt.savefig("../figs/41_parameters_icc_cross_temporal.svg", format="svg", bbox_inches="tight")


# In[ ]:




