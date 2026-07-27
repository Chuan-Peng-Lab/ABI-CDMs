#!/usr/bin/env python
# coding: utf-8

# # Factor Space Visualization: Lab vs. Task Differences
# 
# ## Overview
# 
# This notebook visualizes the distribution of participants across the **four EFA-derived factor dimensions** and compares differences between **tasks** (`task_name`) and **labs** (`author_year`).
# 
# ### Background
# 
# The previous dimensionality-reduction analysis (notebook `42task_difference_analysis.ipynb`) used t-SNE/UMAP to project all model parameters into 2D space and showed that lab (author) variation was larger than task variation. However, that 2D projection lacks direct interpretability because the axes have no cognitive meaning.
# 
# **This notebook addresses that limitation** by working directly in the 4-dimensional EFA factor space:
# 
# | Factor | Type | Cognitive meaning |
# |---|---|---|
# | **Decision Caution** | Shared across tasks | Boundary separation / response conservatism |
# | **Non-decision time** | Shared across tasks | Encoding + motor execution time |
# | **Processing Efficiency** | Conflict-specific | Drift rate / signal quality |
# | **Inhibitory process** | Conflict-specific | Selective attention / conflict control |
# 
# ### Analysis Strategy
# 
# Because the factors come in two pairs that share a cognitive theme, we create **two 2D scatter plots**:
# 
# 1. **Shared factors** (`Decision Caution` × `Non-decision time`) — both factors are common across tasks, so task differences here reflect *general cognitive capacity*.
# 2. **Conflict-specific factors** (`Processing Efficiency` × `Inhibitory process`) — these vary in meaning across tasks, and separation here would reflect *true Conflict-specific mechanisms*.
# 
# Each plot is then faceted by `task_name` and `author_year` to allow visual comparison of lab vs. task effects.
# 
# In addition, **between-group Euclidean distance** (centroid-based) is computed within each factor space to quantify lab vs. task separation quantitatively.

# ## 1. Imports and Setup

# In[1]:


import sys
from nsbi_module.utils_ind_diff import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from itertools import combinations
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
sns.set_style('white')

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── Figure aesthetics ────────────────────────────────────────────────────────
PALETTE_TASK = 'tab20'
PALETTE_LAB  = 'tab20'
ALPHA_SCATTER = 0.35
POINT_SIZE    = 80
LABEL_FONTSIZE = 14
TICK_FONTSIZE  = 11
LEGEND_FONTSIZE = 9

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## 2. Load Data
# 
# We read the pre-processed data file that already contains the four EFA factor scores alongside the raw model parameters and grouping variables.

# In[2]:


DATA_PATH = '43_subj_indices_with_EFA_scores.csv'

df = pd.read_csv(DATA_PATH)

# ── Capitalise group labels for cleaner figure legends ───────────────────────
df['task_name']   = df['task_name'].str.capitalize()
df['author_year'] = df['author_year'].str.capitalize()

# ── Factor column names ──────────────────────────────────────────────────────
FACTOR_CAUTION    = 'Decision Caution'
FACTOR_NDT        = 'Non-decision time'
FACTOR_EFFICIENCY = 'Processing Efficiency'
FACTOR_INHIBITION = 'Inhibitory process'

ALL_FACTORS = [FACTOR_CAUTION, FACTOR_NDT, FACTOR_EFFICIENCY, FACTOR_INHIBITION]

print(f'Dataset shape: {df.shape}')
print(f'Tasks  : {sorted(df["task_name"].unique())}')
print(f'Labs   : {sorted(df["author_year"].unique())}')
print()
print('Factor score summary:')
df[ALL_FACTORS].describe().round(3)


# ## 3. Helper Functions
# 
# All plotting and analysis utilities are defined here so the notebook cells that follow remain concise.

# In[3]:


# ─────────────────────────────────────────────────────────────────────────────
# 3.1  Scatter plot: factor space coloured by a grouping variable
# ─────────────────────────────────────────────────────────────────────────────
def plot_factor_scatter(
    df,
    x_factor,
    y_factor,
    hue_col,
    palette,
    ax,
    show_centroids=True,
    max_points_per_group=None,
    alpha=ALPHA_SCATTER,
    point_size=POINT_SIZE,
    label_fontsize=LABEL_FONTSIZE,
    tick_fontsize=TICK_FONTSIZE,
    legend_fontsize=LEGEND_FONTSIZE,
):
    """
    Draw a scatter plot of two EFA factor scores coloured by *hue_col*.

    Parameters
    ----------
    df                  : pd.DataFrame  — data to plot
    x_factor            : str           — column name for the x-axis factor
    y_factor            : str           — column name for the y-axis factor
    hue_col             : str           — grouping column (e.g. 'task_name' or 'author_year')
    palette             : str or list   — seaborn palette name or explicit colour list
    ax                  : Axes          — matplotlib axes to draw on
    show_centroids      : bool          — whether to overlay group centroids
    max_points_per_group: int or None   — if provided, randomly sample each group in 
                                         hue_col down to this number for the scatter plot.
                                         None means no downsampling. Centroids are always 
                                         calculated from the full original data.

    Returns
    -------
    ax : Axes
    legend_handles : list or None
        Custom legend handles if show_centroids=True, else None
    """
    import pandas as pd
    import matplotlib.lines as mlines

    # 1. Explicitly determine the grouping order to ensure color consistency
    if pd.api.types.is_categorical_dtype(df[hue_col]):
        hue_order = df[hue_col].cat.categories.tolist()
    else:
        hue_order = sorted(df[hue_col].unique())

    # 2. Generate the corresponding color list based on the fixed order
    colours = sns.color_palette(palette, n_colors=len(hue_order))

    # 3. Handle downsampling: create a copy and sample only for the scatter plot
    df_plot = df.copy()
    if max_points_per_group is not None and max_points_per_group > 0:
        df_plot = df_plot.groupby(hue_col, group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), max_points_per_group), random_state=42)
        )

    # 4. Draw scatter plot using the (potentially downsampled) df_plot
    sns.scatterplot(
        data=df_plot,
        x=x_factor,
        y=y_factor,
        hue=hue_col,
        hue_order=hue_order,
        palette=colours,
        alpha=alpha,
        s=point_size,
        ax=ax
    )

    ax.axhline(y=0, color='grey', linestyle='--', linewidth=3, alpha=0.5)
    ax.axvline(x=0, color='grey', linestyle='--', linewidth=3, alpha=0.5)

    legend_handles = None
    if show_centroids:
        # 5. Calculate centroids using the FULL original df (unaffected by downsampling)
        centroids = df.groupby(hue_col)[[x_factor, y_factor]].mean()

        # 6. Force the centroids index to match our defined hue_order exactly
        centroids = centroids.reindex(hue_order)

        legend_handles = []
        # 7. Iterate: centroids order and colours list are now 100% aligned
        for (group, row), colour in zip(centroids.iterrows(), colours):
            ax.scatter(
                row[x_factor], row[y_factor],
                marker='D', s=90, color=colour,
                edgecolors='black', linewidths=0.8, zorder=5
            )
            legend_handles.append(
                mlines.Line2D([], [], marker='D', linestyle='None', 
                              markerfacecolor=colour, markeredgecolor='black',
                              markeredgewidth=0.8, markersize=9, label=str(group))
            )
        # Remove the auto-generated seaborn legend (semi-transparent dots)
        # and replace it with centroid-colour solid diamond markers.
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.legend(handles=legend_handles, fontsize=legend_fontsize,
                  framealpha=0.6, edgecolor='none')
    else:
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title('')
            for text in legend.get_texts():
                text.set_fontsize(legend_fontsize)

    ax.set_xlabel(x_factor, fontsize=label_fontsize)
    ax.set_ylabel(y_factor, fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)

    sns.despine(ax=ax)
    return ax, legend_handles


# ─────────────────────────────────────────────────────────────────────────────
# 3.2  Compute pairwise centroid Euclidean distances for a grouping variable
# ─────────────────────────────────────────────────────────────────────────────

def compute_centroid_distances(df, x_factor, y_factor, group_col):
    """
    Calculate pairwise Euclidean distances between group centroids in a 2D
    factor space.

    Parameters
    ----------
    df         : pd.DataFrame — source data
    x_factor   : str          — column for the x-axis factor
    y_factor   : str          — column for the y-axis factor
    group_col  : str          — grouping column

    Returns
    -------
    pd.DataFrame with columns ['Grouping', 'Group_A', 'Group_B', 'Distance']
    """
    centroids  = df.groupby(group_col)[[x_factor, y_factor]].mean()
    labels     = centroids.index.tolist()
    records    = []
    for (i1, lbl1), (i2, lbl2) in combinations(enumerate(labels), 2):
        p1   = centroids.iloc[i1].values
        p2   = centroids.iloc[i2].values
        dist = np.linalg.norm(p1 - p2)
        records.append({
            'Grouping': group_col,
            'Group_A' : lbl1,
            'Group_B' : lbl2,
            'Distance': dist,
        })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 3.3  Distance distribution box-strip plot with statistical annotation
# ─────────────────────────────────────────────────────────────────────────────

def make_factor_pair_figure(
    df,
    x_factor,
    y_factor,
    title='',
    palette_task=PALETTE_TASK,
    palette_lab=PALETTE_LAB,
    figsize=(16, 5),
    save_path=None,
):
    """
    Build a 3-panel figure:
      Panel A — scatter coloured by task_name
      Panel B — scatter coloured by author_year
      Panel C — distance distribution box-strip with significance bracket

    Parameters
    ----------
    df         : pd.DataFrame  — EFA-enriched dataset
    x_factor   : str           — column for x-axis
    y_factor   : str           — column for y-axis
    title      : str           — overall figure suptitle
    palette_task : str         — palette for task colouring
    palette_lab  : str         — palette for lab colouring
    figsize    : tuple         — (width, height)
    save_path  : str or None   — if given, saves as SVG

    Returns
    -------
    fig, (ax_task, ax_lab, ax_dist)
    """
    from matplotlib.gridspec import GridSpec
    sns.set_style('white')

    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(1, 4, figure=fig, width_ratios=[1.1, 1.35, 0.15, 0.75],
                   wspace=0.35)
    ax_task = fig.add_subplot(gs[0, 0])
    ax_lab  = fig.add_subplot(gs[0, 1])
    ax_dist = fig.add_subplot(gs[0, 3])

    # ── Panel A: coloured by task ──────────────────────────────────────────
    ax_task, task_legend_handles = plot_factor_scatter(
        df, x_factor, y_factor,
        hue_col='task_name', palette=palette_task,
        ax=ax_task, show_centroids=True,
    )
    ax_task.legend(
        handles=task_legend_handles,
        loc='best', fontsize=LEGEND_FONTSIZE,
        framealpha=0.6, edgecolor='none',
    )
    ax_task.text(-0.18, 1.04, 'A', transform=ax_task.transAxes,
                 fontsize=20, fontweight='bold', va='bottom')

    # ── Panel B: coloured by lab ───────────────────────────────────────────
    ax_lab, lab_legend_handles = plot_factor_scatter(
        df, x_factor, y_factor,
        hue_col='author_year', palette=palette_lab,
        ax=ax_lab, show_centroids=True,
    )
    # Remove the legend from ax_lab and create figure-level legend with custom handles
    if ax_lab.get_legend() is not None:
        ax_lab.get_legend().remove()
    bbox = ax_lab.get_position()
    fig.legend(
        handles=lab_legend_handles,
        loc='center left',
        bbox_to_anchor=(bbox.x1 + 0.01, bbox.y0 + bbox.height / 2),
        fontsize=LEGEND_FONTSIZE, framealpha=0.6,
    )
    ax_lab.text(-0.18, 1.04, 'B', transform=ax_lab.transAxes,
                fontsize=20, fontweight='bold', va='bottom')

    # ── Panel C: centroid distance box-plot ────────────────────────────────
    dist_task = compute_centroid_distances(df, x_factor, y_factor, 'task_name')
    dist_lab  = compute_centroid_distances(df, x_factor, y_factor, 'author_year')
    dist_all  = pd.concat([dist_task, dist_lab], ignore_index=True)

    plot_distance_comparison(dist_all, ax=ax_dist)
    ax_dist.text(-0.25, 1.04, 'C', transform=ax_dist.transAxes,
                 fontsize=20, fontweight='bold', va='bottom')

    if title:
        fig.suptitle(title, fontsize=15, y=1.02, fontweight='bold')

    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        print(f'Figure saved to: {save_path}')

    return fig, (ax_task, ax_lab, ax_dist)


# ## 4. Plot 1 — Shared Factors: Decision Caution × Non-decision Time
# 
# Both **Decision Caution** (boundary separation, linked to speed-accuracy trade-off) and **Non-decision time** (perceptual encoding + motor response latency) are factors that are shared across cognitive tasks — they reflect general, task-agnostic cognitive tendencies.
# 
# If lab effects dominate here (Panel C shows larger centroid distances for *Across Labs* than *Across Tasks*), it would suggest that these general cognitive characteristics are more strongly shaped by between-study methodology (e.g., inter-stimulus intervals, practice effects, participant recruitment) than by the specific cognitive manipulation being tested.

# In[4]:


fig1, axes1 = make_factor_pair_figure(
    df,
    x_factor=FACTOR_CAUTION,
    y_factor=FACTOR_NDT,
    title='Factor Space 1: Shared Factors — Decision Caution × Non-decision Time',
    figsize=(17, 5),
    # save_path='figs/43_factor_space_shared.svg',  # uncomment to save
)
plt.tight_layout()


# ### 4.1 Descriptive Statistics for Shared Factors

# In[5]:


def summarise_factor_pair(df, x_factor, y_factor, group_cols=('task_name', 'author_year')):
    """
    Print mean centroid distances and Mann-Whitney U test p-value for two factors.

    Parameters
    ----------
    df          : pd.DataFrame
    x_factor    : str  — column name for factor 1
    y_factor    : str  — column for factor 2
    group_cols  : tuple of str — grouping variables to compare
    """
    print(f'Factor pair: [{x_factor}] vs [{y_factor}]')
    print('=' * 60)
    dist_records = []
    for col in group_cols:
        dist_df = compute_centroid_distances(df, x_factor, y_factor, col)
        label   = 'Across Tasks' if col == 'task_name' else 'Across Labs'
        mean_d  = dist_df['Distance'].mean()
        std_d   = dist_df['Distance'].std()
        print(f'{label:15s}  n_pairs={len(dist_df):3d}  '
              f'mean={mean_d:.4f}  std={std_d:.4f}')
        dist_records.append(dist_df)

    if len(dist_records) == 2:
        g1, g2 = [d['Distance'] for d in dist_records]
        stat, p = stats.mannwhitneyu(g1, g2)
        print(f'\nMann-Whitney U: stat={stat:.1f}  p={p:.4e}')
    print()


summarise_factor_pair(df, FACTOR_CAUTION, FACTOR_NDT)


# ## 5. Plot 2 — Conflict-specific Factors: Processing Efficiency × Inhibitory Process
# 
# **Processing Efficiency** (roughly analogous to drift rate — how fast evidence accumulates toward the correct response) and **Inhibitory process** (selective attention, conflict-resolution efficiency) are factors that, while extractable from the model parameters, embed Conflict-specific signal more strongly.
# 
# If tasks cluster clearly in this panel (**Panel C** shows *larger Across-Tasks distances*), it would support the hypothesis that these factors capture genuine task differences — i.e., tasks differ in the specific cognitive sub-process they demand, rather than in general capacity.

# In[6]:


fig2, axes2 = make_factor_pair_figure(
    df,
    x_factor=FACTOR_EFFICIENCY,
    y_factor=FACTOR_INHIBITION,
    title='Factor Space 2: Conflict-specific Factors — Processing Efficiency × Inhibitory Process',
    figsize=(17, 5),
    # save_path='figs/43_factor_space_taskspecific.svg',  # uncomment to save
)
plt.tight_layout()


# In[7]:


summarise_factor_pair(df, FACTOR_EFFICIENCY, FACTOR_INHIBITION)


# ## 6. Plot Combination

# In[9]:


from matplotlib.gridspec import GridSpec
PALETTE = sns.color_palette("husl",20)
PALETTE_TASK = [PALETTE[0], PALETTE[12], PALETTE[14]]
PALETTE_LAB = sns.color_palette("Set2",10)[1:9]


# ─────────────────────────────────────────────────────────────────────────────
# Combined Figure: Shared Factors & Conflict-specific Factors
# ─────────────────────────────────────────────────────────────────────────────

def make_combined_factor_figure(
    df,
    x_factor1, y_factor1, title1,
    x_factor2, y_factor2, title2,
    palette_task=PALETTE_TASK,
    palette_lab=PALETTE_LAB,
    max_points_per_group = None,
    figsize=(15, 10),
    save_path=None,
):
    """
    Build a 6-panel figure (2 rows x 3 columns):
      Row 1 (Pair 1): Panel A (Task scatter), Panel B (Lab scatter), Panel C (Distance)
      Row 2 (Pair 2): Panel D (Task scatter), Panel E (Lab scatter), Panel F (Distance)
    Shared legends for Tasks (A & D) and Labs (B & E) are extracted and placed at the bottom.
    """
    sns.set_style("white")

    fig = plt.figure(figsize=figsize)

    gs = GridSpec(
        2, 3, figure=fig, 
        width_ratios=[0.37, 0.37, 0.26], 
        # wspace=0.35, 
        hspace=0.55
        )

    # Initialize subplots
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[0, 2])

    ax_D = fig.add_subplot(gs[1, 0])
    ax_E = fig.add_subplot(gs[1, 1])
    ax_F = fig.add_subplot(gs[1, 2])

    # =========================================================================
    # ROW 1: First Factor Pair (e.g., Caution x NDT)
    # =========================================================================

    # Panel A: Task Scatter
    ax_A, task_legend_handles = plot_factor_scatter(
        df, x_factor1, y_factor1,
        hue_col='task_name', palette=palette_task,
        max_points_per_group = max_points_per_group,
        ax=ax_A, show_centroids=True,
    )
    ax_A.set_title(title1, loc='left', fontsize=14, fontweight='bold', pad=50)
    ax_A.text(-0.18, 1.04, 'A', transform=ax_A.transAxes, fontsize=20, fontweight='bold', va='bottom')

    # Panel B: Lab Scatter
    ax_B, lab_legend_handles = plot_factor_scatter(
        df, x_factor1, y_factor1,
        hue_col='author_year', palette=palette_lab,
        max_points_per_group = max_points_per_group,
        ax=ax_B, show_centroids=True,
    )
    ax_B.text(-0.18, 1.04, 'B', transform=ax_B.transAxes, fontsize=20, fontweight='bold', va='bottom')

    # Panel C: Distance Boxplot
    dist_task1 = compute_centroid_distances(df, x_factor1, y_factor1, 'task_name')
    dist_lab1  = compute_centroid_distances(df, x_factor1, y_factor1, 'author_year')
    dist_all1  = pd.concat([dist_task1, dist_lab1], ignore_index=True)
    plot_distance_comparison(dist_all1, ax=ax_C)
    ax_C.text(-0.25, 1.04, 'C', transform=ax_C.transAxes, fontsize=20, fontweight='bold', va='bottom')


    # =========================================================================
    # ROW 2: Second Factor Pair (e.g., Efficiency x Inhibition)
    # =========================================================================

    # Panel D: Task Scatter
    ax_D, _ = plot_factor_scatter(
        df, x_factor2, y_factor2,
        hue_col='task_name', palette=palette_task,
        max_points_per_group = max_points_per_group,
        ax=ax_D, show_centroids=True,
    )
    ax_D.set_title(title2, loc='left', fontsize=14, fontweight='bold', pad=50)
    ax_D.text(-0.18, 1.04, 'D', transform=ax_D.transAxes, fontsize=20, fontweight='bold', va='bottom')

    # Panel E: Lab Scatter
    ax_E, _ = plot_factor_scatter(
        df, x_factor2, y_factor2,
        hue_col='author_year', palette=palette_lab,
        max_points_per_group = max_points_per_group,
        ax=ax_E, show_centroids=True,
    )
    ax_E.text(-0.18, 1.04, 'E', transform=ax_E.transAxes, fontsize=20, fontweight='bold', va='bottom')

    # Panel F: Distance Boxplot
    dist_task2 = compute_centroid_distances(df, x_factor2, y_factor2, 'task_name')
    dist_lab2  = compute_centroid_distances(df, x_factor2, y_factor2, 'author_year')
    dist_all2  = pd.concat([dist_task2, dist_lab2], ignore_index=True)
    plot_distance_comparison(dist_all2, ax=ax_F)
    ax_F.text(-0.25, 1.04, 'F', transform=ax_F.transAxes, fontsize=20, fontweight='bold', va='bottom')


    # =========================================================================
    # Legend Extraction & Formatting
    # =========================================================================

    # Remove any existing local legends from the scatter plots to avoid clutter
    for ax in [ax_A, ax_B, ax_D, ax_E]:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # Create space at the bottom of the figure for the unified legends
    fig.subplots_adjust(bottom=0.18)

    # Add global Task legend below panels A and D
    fig.legend(
        handles=task_legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.25, 0.11),  # Adjust X/Y to position properly below first column
        ncol=len(task_legend_handles), # Put items in a single row
        fontsize=LEGEND_FONTSIZE,
        framealpha=0.6,
        title='Tasks',
        title_fontsize=LEGEND_FONTSIZE + 1
    )

    # Add global Lab legend below panels B and E
    fig.legend(
        handles=lab_legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.65, 0.11), # Adjust X/Y to position properly below second column
        ncol=4,                      # Wrap into rows of 4 if there are many labs
        fontsize=LEGEND_FONTSIZE,
        framealpha=0.6,
        title='Labs / Studies',
        title_fontsize=LEGEND_FONTSIZE + 1
    )

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        print(f'Combined figure saved to: {save_path}')

    return fig, (ax_A, ax_B, ax_C, ax_D, ax_E, ax_F)


# ─────────────────────────────────────────────────────────────────────────────
# Execute the combined figure plot
# ─────────────────────────────────────────────────────────────────────────────

fig_combined, axes_combined = make_combined_factor_figure(
    df=df,
    # First Row Settings
    x_factor1=FACTOR_CAUTION,
    y_factor1=FACTOR_NDT,
    title1='Factor Space 1: Model-shared Factors — Decision Caution × Non-decision Time',

    # Second Row Settings
    x_factor2=FACTOR_EFFICIENCY,
    y_factor2=FACTOR_INHIBITION,
    title2='Factor Space 2: Model-specific Factors — Processing Efficiency × Inhibitory Process',

    # Global Settings
    figsize=(13, 11),  # Height increased to 11 to accommodate 2 rows + bottom legend
    save_path='../figs/43_factor_space_combined.svg',  
)

# plt.tight_layout()

