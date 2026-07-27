#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import re
import warnings

warnings.filterwarnings('ignore')

# --- Configuration & Setup ---

# Input file path
input_file = '../03_fitting/23model_prediction_indices.csv'

# Metrics to compare in the columns of the grid plot
comparison_metrics = ['RMSE', 'g_square', 'aBIC']

# Model definitions and color palette
MODELS = ['DDM', 'SSP', 'DMC', 'DSTP']
COLORS = ['#9dd25b', '#6fc96f', '#75cfb6', '#4dc3eb']
model_palette = dict(zip(MODELS, COLORS))

# The order used for stacking colors (bottom to top) and sorting participants (left to right)
STACK_AND_SORT_ORDER = ['DSTP', 'DMC', 'SSP', 'DDM']


# In[2]:


def load_and_preprocess_data(filepath, metrics):
    """
    Loads model prediction data from a CSV file and performs basic preprocessing.

    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing model prediction indices.
    metrics : list of str
        List of metric columns to ensure are present and non-null.

    Returns:
    --------
    pd.DataFrame
        Cleaned and preprocessed DataFrame.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)

    # Convert essential columns to string types
    df['task_id'] = df['task_id'].astype(str)
    df['model'] = df['model'].astype(str)
    df['subject_id'] = df['subject_id'].astype(str)

    # Ensure model names are capitalized
    df['model'] = df['model'].str.upper()

    # Drop rows with missing values in crucial columns
    df = df.dropna(subset=['task_id', 'model', 'subject_id'] + metrics)

    print(f"Loaded {len(df)} rows.")
    print(f"Found {len(df['task_id'].unique())} distinct tasks and {len(df['model'].unique())} models.")
    return df

clean_df = load_and_preprocess_data(input_file, comparison_metrics)


# In[3]:


def convert_metrics_to_weights(df, metric_col):
    """
    Converts raw model metrics into weights (probabilities) that sum to 1 per subject per task.

    For information criteria (e.g., aBIC), Schwarz weights are calculated:
    w_i = exp(-0.5 * (metric_i - metric_min)) / Sum[exp(-0.5 * (metric_j - metric_min))]

    For loss functions (e.g., RMSE, chi-square), an inverse-value heuristic is used:
    w_i = (1/metric_i) / Sum[1/metric_j]. Lower metric values result in higher weights.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the raw metrics.
    metric_col : str
        The name of the metric column to convert.

    Returns:
    --------
    pd.DataFrame
        DataFrame with an additional '{metric_col}_weight' column.
    """
    weight_col = f"{metric_col}_weight"

    # Group by task and subject to find the best fit within each context
    grouped = df.groupby(['task_id', 'subject_id'])
    min_values = grouped[metric_col].transform('min')

    if metric_col == 'aBIC':
        # Schwarz weights logic for IC metrics
        df[weight_col] = np.exp(-0.5 * (df[metric_col] - min_values))
    elif metric_col in ['RMSE', 'g_square']:
        # Inverse logic for loss functions (heuristic for visualization)
        # Handles potential division by zero by using a small epsilon
        df[weight_col] = 1.0 / np.where(df[metric_col] == 0, 1e-10, df[metric_col])

    # Normalize weights so they sum to 1.0 within each task-subject group
    group_sums = df.groupby(['task_id', 'subject_id'])[weight_col].transform('sum')
    df[weight_col] = df[weight_col] / group_sums
    df[weight_col] = df[weight_col].fillna(0)

    return df

def create_sorted_pivot(task_data, weight_col, model_order):
    """
    Pivots data to a wide format and sorts subjects to create a smooth 'waterfall' effect.

    Parameters:
    -----------
    task_data : pd.DataFrame
        DataFrame containing model weights for a single task.
    weight_col : str
        The name of the weight column to pivot.
    model_order : list of str
        The desired order of models for columns (stacking) and sorting priority.

    Returns:
    --------
    pd.DataFrame
        Pivoted and sorted DataFrame with subjects as index and models as columns.
    """
    # Pivot: index=subject, columns=model, values=weight
    pivoted = task_data.pivot(index='subject_id', columns='model', values=weight_col)

    # Ensure all models are present in the columns
    for model in model_order:
        if model not in pivoted.columns:
            pivoted[model] = 0.0

    # Reorder columns to determine the vertical stacking order (bottom to top)
    pivoted = pivoted[model_order]

    # Sort by model weights in descending order to create the horizontal waterfall effect
    pivoted = pivoted.sort_values(by=model_order, ascending=[False] * len(model_order))

    return pivoted

def format_task_label(task_str):
    """
    Formats a raw task ID string into a readable multiline label.
    Example: 'ulrich2015flanker' -> 'Ulrich2015\nFlanker'

    Parameters:
    -----------
    task_str : str
        The raw task ID string.

    Returns:
    --------
    str
        Formatted task label.
    """
    match = re.match(r'([a-zA-Z]+)(\d{4})([a-zA-Z]+)', task_str)
    if match:
        author_part, year_part, task_part = match.groups()
        formatted_author = author_part.capitalize()
        formatted_task = ''.join(word.capitalize() for word in re.findall('[a-zA-Z][^A-Z]*', task_part))
        return f"{formatted_author}{year_part}\n{formatted_task}"
    return task_str


# In[4]:


def generate_complex_plot(df, metrics, palette, sort_order, single_figsize=(5, 2), datasets_group_col='task_id'):
    """
    Generates a multi-panel grid of stacked bar plots comparing model performance.

    Parameters:
    -----------
    df : pd.DataFrame
        The preprocessed DataFrame containing model metrics.
    metrics : list of str
        Metrics to compare (columns in the plot).
    palette : dict
        Mapping from model names to colors.
    sort_order : list of str
        The order of models used for vertical stacking and horizontal sorting.
    single_figsize : tuple, optional
        Size (width, height) of a single subplot panel.
    datasets_group_col : str, optional
        The column name used to group datasets (rows in the plot).
    """
    unique_datasets = df[datasets_group_col].unique()
    num_datasets = len(unique_datasets)
    num_cols = len(metrics)

    fig, axes = plt.subplots(num_datasets, num_cols,
                             figsize=(single_figsize[0] * num_cols, single_figsize[1] * num_datasets),
                             constrained_layout=True, sharey=True)

    if num_datasets == 1:
        axes = axes.reshape(1, num_cols)

    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 12})

    for row_idx, task_id in enumerate(unique_datasets):
        task_data_raw = df[df[datasets_group_col] == task_id].copy()

        for col_idx, metric_col in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            weight_col = f"{metric_col}_weight"

            # 1. Convert raw metrics to weights
            task_data_w = convert_metrics_to_weights(task_data_raw.copy(), metric_col)

            # 2. Pivot and sort using the specific order to maintain the waterfall effect
            pivoted_data = create_sorted_pivot(task_data_w, weight_col, sort_order)

            # 3. Plot stacked bars (colors are assigned based on column order)
            current_colors = [palette[m] for m in pivoted_data.columns]
            pivoted_data.plot(kind='bar', stacked=True, color=current_colors,
                              ax=ax, width=1.0, edgecolor=None, legend=False)

            # 4. Axes formatting
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.5, 1])
            ax.set_yticklabels(['0', '0.5', '1'], fontsize=11)
            ax.yaxis.set_ticks_position('left')
            ax.grid(axis='y', linestyle='-', color='#eeeeee', linewidth=0.5)

            num_participants = len(pivoted_data)
            ax.set_xticks([0, num_participants - 1])
            ax.set_xticklabels(['0', f'{num_participants}'], rotation=0, fontsize=11)
            ax.tick_params(axis='x', which='both', length=3, direction='out')

            # 5. Annotation (Mean Probabilities)
            # We keep a consistent order for annotations (e.g., matching the legend order)
            mean_probs = task_data_w.groupby('model')[weight_col].mean().reindex(list(palette.keys())).to_dict()
            annotation_str = "  ".join([f"${m}$: {p:.2f}" for m, p in mean_probs.items()])

            ax.text(0.98, 1.05, annotation_str,
                    transform=ax.transAxes, fontsize=12, fontweight='bold',
                    va='bottom', ha='right', color='black')

            # 6. Titles and Labels
            if col_idx == 0:
                ax.set_ylabel(format_task_label(task_id), fontsize=11, fontweight='bold', rotation=90)

            if row_idx == 0:
                ax.set_title(metric_col.replace('_', ' '), fontsize=16, fontweight='bold', pad=26)

            if row_idx == num_datasets - 1:
                ax.set_xlabel("Participant", fontsize=14)

    # Add overall legend matching the initial color definition
    legend_patches = [mpatches.Patch(color=color, label=label) for label, color in palette.items()]
    fig.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, 1.01),
               ncol=len(palette), fontsize=12, frameon=False)

# Execution
generate_complex_plot(clean_df, comparison_metrics, model_palette, STACK_AND_SORT_ORDER)
plt.savefig("../figs/32_cognitive_model_weights.svg", format='svg', bbox_inches='tight')

