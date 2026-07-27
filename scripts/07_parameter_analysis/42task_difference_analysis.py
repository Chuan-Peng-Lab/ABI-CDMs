#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
import rsatoolbox
import warnings
from itertools import combinations
from scipy import stats
warnings.filterwarnings('ignore')

import sys
from nsbi_module.utils_ind_diff import *


sns.set_style("white")

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


def wrap_parameter(param):
    """
    Wrap parameter name in $...$ format and add {} around subscripts for LaTeX.
    """
    param_str = str(param)
    if param_str.startswith('$') and param_str.endswith('$'):
        return param_str

    if '_' in param_str:
        parts = param_str.split('_', 1)
        formatted_param = f'{parts[0]}_{{{parts[1]}}}'
        return f'${formatted_param}$'
    else:
        return f'${param_str}$'

def preprocess_data(df: pd.DataFrame):
    """
    Standardize numeric data and filter out non-feature columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude common non-feature columns
    for col in ['subject_id', 'tsne_1', 'tsne_2', 'umap_1', 'umap_2']:
        if col in numeric_cols:
            numeric_cols.remove(col)

    # Handle missing values by filling with column mean
    df_numeric = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Standardize numeric variables
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_numeric),
        columns=numeric_cols,
        index=df.index
    )

    return df_scaled

def get_cluster_projection(df: pd.DataFrame, method='t-SNE', random_state=42):
    """
    Perform dimensionality reduction and return a DataFrame with coordinate columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    method : str
        Dimensionality reduction method ('t-SNE' or 'UMAP').
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        A copy of the original DataFrame with added projection coordinates 
        and LaTeX-formatted column names.
    """
    # 1. Scaling
    df_scaled = preprocess_data(df)

    # 2. Reduction
    if method == 't-SNE':
        print("Computing t-SNE projection...")
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=30)
    elif method == 'UMAP':
        print("Computing UMAP projection...")
        reducer = umap.UMAP(n_components=2, random_state=random_state)
    else:
        raise ValueError("Method must be 't-SNE' or 'UMAP'")

    projection = reducer.fit_transform(df_scaled)

    # 3. Create Result DataFrame
    res_df = df.copy()

    # Apply LaTeX formatting to specific parameter columns for better plot labels later
    columns_to_format = ['rt_avg', 'rt_cost', 'acc_avg', 'error_cost', 'delta_slope']
    res_df.rename(columns={col: wrap_parameter(col) for col in columns_to_format if col in res_df.columns}, inplace=True)

    # Add coordinates
    col_prefix = method.lower().replace('-', '')
    res_df[f'{col_prefix}_1'] = projection[:, 0]
    res_df[f'{col_prefix}_2'] = projection[:, 1]

    # 3. author_year and task_name are transformed to first character with capital letter
    res_df['author_year'] = res_df['author_year'].apply(lambda x: x[0].upper() + x[1:])
    res_df['task_name'] = res_df['task_name'].apply(lambda x: x[0].upper() + x[1:])

    return res_df

def analyze_group_distances(df, method='t-SNE', group_cols=['task_name', 'author_year']):
    """
    Calculate centroids and pairwise distances, including labels for the pairs compared.

    Returns
    -------
    dict, pd.DataFrame
        Detailed results and a DataFrame containing 'Grouping', 'Pair_A', 'Pair_B', and 'Distance'.
    """
    col_prefix = method.lower().replace('-', '')
    coords_cols = [f'{col_prefix}_1', f'{col_prefix}_2']

    results = {}
    distance_data = []

    for col in group_cols:
        # 1. Calculate Centroids
        centroids = df.groupby(col)[coords_cols].mean()
        group_labels = centroids.index.tolist()

        # 2. Calculate Pairwise distances with labels
        # We iterate through combinations to keep track of which groups are being compared
        dist_list = []
        for (idx1, label1), (idx2, label2) in combinations(enumerate(group_labels), 2):
            p1 = centroids.iloc[idx1].values
            p2 = centroids.iloc[idx2].values

            # Euclidean distance: ||p1 - p2||
            dist = np.linalg.norm(p1 - p2)

            dist_list.append(dist)
            distance_data.append({
                'Grouping': col,
                'Group_A': label1,
                'Group_B': label2,
                'Pair': f"{label1} vs {label2}",
                'Distance': dist
            })

        results[col] = {
            'centroids': centroids,
            'distances': np.array(dist_list),
            'mean_dist': np.mean(dist_list),
            'std_dist': np.std(dist_list)
        }

    return results, pd.DataFrame(distance_data)

def plot_cluster_comparison(df_with_coords, method='t-SNE', facet_groupby=['task_name', 'author_year'], 
                            figsize=(18, 6), alpha=0.3, save_path=None, label_fontsize=17, palette='tab20', ax=None, first_legend_pos='upper right', first_legend_bbox=None, 
                            legend_fontsize = None, second_legend_pos='center left'):
    """
    Generate side-by-side comparison plots for cluster projections.

    Parameters
    ----------
    df_with_coords : pd.DataFrame
        DataFrame containing the projection columns.
    method : str
        Method name used for column indexing ('t-SNE' or 'UMAP').
    facet_groupby : list
        List of column names to create subplots for.
    figsize : tuple
        Figure dimensions.
    alpha : float
        Transparency of scatter points.
    save_path : str, optional
        Path to save the figure.
    ax : list or matplotlib.axes.Axes, optional
        Axis or list of axes to plot on.

    Returns
    -------
    fig, axes
    """
    n_groups = len(facet_groupby)

    # Handle axes input
    if ax is not None:
        if hasattr(ax, '__iter__'):
            # ax is a list/array of axes
            axes = ax if len(ax) >= n_groups else [ax] if n_groups == 1 else list(ax) + [None]*(n_groups-len(ax))
        else:
            # ax is a single axis object
            axes = [ax] if n_groups == 1 else [ax] + [None]*(n_groups-1)
        fig = None 
    else:
        # Create new figure and axes if none provided
        fig, axes = plt.subplots(1, n_groups, figsize=figsize)
        if n_groups == 1:
            axes = [axes]

    # Plot each group
    for i, group_col in enumerate(facet_groupby):
        target_ax = axes[i]

        # If axis is None (in case of mismatch), create on the fly (rare case)
        if target_ax is None:
            if fig is None:
                fig, target_ax = plt.subplots(figsize=figsize[:2])
            else:
                target_ax = fig.add_subplot(1, n_groups, i+1)
            axes[i] = target_ax

        # Call the single plotting logic (Assuming plot_single_cluster exists or inline logic)
        col_prefix = method.lower().replace('-', '')
        x_col, y_col = f'{col_prefix}_1', f'{col_prefix}_2'
        sns.scatterplot(
            data=df_with_coords, x=x_col, y=y_col, 
            hue=group_col, ax=target_ax, alpha=alpha, palette=palette
        )
        # target_ax.set_title(f'Colored by {group_col.replace("_", " ").title()}')
        target_ax.set_xlabel(f'{method} 1', fontsize=label_fontsize)
        target_ax.set_ylabel(f'{method} 2', fontsize=label_fontsize)

    legend_fontsize = label_fontsize if legend_fontsize is None else legend_fontsize
    axes[0].legend(loc=first_legend_pos, bbox_to_anchor = first_legend_bbox, fontsize=legend_fontsize-8)
    axes[1].legend(loc=second_legend_pos, bbox_to_anchor=(1, 0.5), fontsize=legend_fontsize-10)

    if fig is not None and save_path:
        plt.tight_layout()
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        print(f"Plot successfully saved to: {save_path}")

    return fig, axes

def plot_combined_panel(df_with_coords, dist_df, save_path=None, figsize=(12, 4)):
    """
    Combine cluster comparison and distance comparison into a single figure.

    Layout: 
    - Panel A: Two Cluster plots (t-SNE/UMAP)
    - Panel B: One Distance comparison Boxplot

    Parameters
    ----------
    df_with_coords : pd.DataFrame
        Data for cluster plots.
    dist_df : pd.DataFrame
        Data for distance boxplot.
    save_path : str, optional
        Path to save the combined figure.
    figsize : tuple
        Size of the total figure.
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    # 1. Create a figure with a grid layout (1 row, 3 columns)
    # Using width_ratios to make the boxplot slightly narrower (0.8) for aesthetics
    fig = plt.figure(figsize=(12, 4))
    gs_main = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 0.15, 0.8])    
    gs_left = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[0, :2], wspace=0.3)

    ax1 = fig.add_subplot(gs_left[0, 0])
    ax2 = fig.add_subplot(gs_left[0, 1])
    ax3 = fig.add_subplot(gs_main[0, 3])

    # 2. Plot Panel A (Cluster Comparison)
    # Pass the first two axes (axes[0] and axes[1]) to the cluster function
    plot_cluster_comparison(
        df_with_coords, 
        method='t-SNE', 
        facet_groupby=['task_name', 'author_year'],
        ax=[ax1, ax2], 
        alpha=0.5
    )
    handles, labels = ax2.get_legend_handles_labels()
    legend_bbox = ax2.get_position()
    legend_x = legend_bbox.x1
    legend_y = legend_bbox.y0 + legend_bbox.height/2 
    fig.legend(handles, labels, 
            loc='center left', 
            bbox_to_anchor=(legend_x, legend_y),  
            fontsize=7)
    ax2.get_legend().remove()

    # 3. Plot Panel B (Distance Comparison)
    # Pass the third axis (axes[2]) to the distance function
    plot_distance_comparison(
        dist_df, 
        ax=ax3,
        stat_test='mannwhitneyu'
    )

    # 4. Add Labels A and B
    # Label "A" placed on the first plot (covers the first two visually)
    ax1.text(-0.2, 1.05, 'A', transform=ax1.transAxes, 
                 fontsize=24, fontweight='bold', va='bottom', ha='right')

    # Label "B" placed on the third plot
    ax3.text(-0.2, 1.05, 'B', transform=ax3.transAxes, 
                 fontsize=24, fontweight='bold', va='bottom', ha='right')

    # 5. Final Adjustments
    plt.tight_layout()

    # 6. Save
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        print(f"Combined figure saved to: {save_path}")

    plt.show()


# In[3]:


indices_by_subj = pd.read_csv("23subj_indices_across_models_and_tasks.csv")
indices_by_subj.head()


# ## Cluster plots

# To analyze the distances between groups in the t-SNE space and compare the variability of `task_name` versus `author_year`, we should calculate the **centroids** (mean positions) of each group and then compute the **pairwise Euclidean distances** between these centroids.
# 
# To support your hypothesis that **author variation is greater than task variation**, look for the following in the output:
# 
# 1. **Mean Pairwise Distance:** If the mean distance between `author_year` centroids is significantly higher than the mean distance between `task_name` centroids, it indicates that "Authors" (studies) occupy more distinct/distant regions of the latent space than the "Tasks" themselves.
# 2. **Distance Distribution (Boxplot):**
# * If the box for `author_year` is positioned higher on the Y-axis than `task_name`, it proves that different studies are more "dissimilar" to each other than different tasks are.
# * This often suggests a **"Lab Effect"** or "Study Effect," where the methodology or population of a specific author influences the data more than the nature of the cognitive task.
# 
# 
# 3. **Mathematical Note:** The distance calculated is the  norm between the mean coordinates  of group  and group :

# In[4]:


# df_with_coords[["task_id","author_year","task_name","tsne_1","tsne_2"]]


# In[5]:


# df_with_coords = get_cluster_projection(indices_by_subj, method='UMAP')
df_with_coords = get_cluster_projection(indices_by_subj, method='t-SNE')

dist_results, dist_df = analyze_group_distances(df_with_coords, method='t-SNE')

for group, data in dist_results.items():
    print(f"--- Analysis for {group} ---")
    print(f"Number of groups: {len(data['centroids'])}")
    print(f"Mean Pairwise Distance: {data['mean_dist']:.4f}")
    print(f"Distance Std Dev: {data['std_dist']:.4f}\n")

dist_df.sort_values(by='Distance', ascending=False).head(10)


# In[10]:


sns.set_style("white")
plot_cluster_comparison(
    df_with_coords, 
    method='t-SNE', 
    facet_groupby=['task_name', 'author_year'],
    figsize=(11.7, 3.5),
    alpha=0.4,      
    first_legend_pos = "center right",
    first_legend_bbox = (1.52, 0.5),
    legend_fontsize = 23,
)
plt.tight_layout()
plt.subplots_adjust(wspace=0.8)
plt.savefig('../figs/42_cluster_plot_tSNE.svg', format='svg', bbox_inches='tight')


# In[7]:


# plot_cluster_comparison(
#     df_with_coords, 
#     method='t-SNE', 
#     facet_groupby=['task_name', 'author_year'],
#     figsize=(10.5, 3.5),
#     alpha=0.5,        
#     save_path='../figs/42_cluster_plot_tSNE.svg'
# )

# plot_distance_comparison(dist_df)

plot_combined_panel(
    df_with_coords=df_with_coords, 
    dist_df=dist_df, 
    save_path='../figs/42task_variance_tSNE_cluster.svg'
)


# ## RSA analysis

# In[36]:


from scipy.stats import spearmanr

def extract_fitted_datasets(indices_by_subj, groups="task_id"):
    """
    Extract datasets with suffix '_fitted_by_subj' and construct rsatoolbox.data.Dataset list.
    The descriptor is the prefix before '_fitted_by_subj'.
    """
    data_list = []
    names = []
    for key, df in indices_by_subj.groupby(groups):
        df = df.copy().drop(columns=["subject_id", 'task_id', 'author_year','task_name'])
        dataset = rsatoolbox.data.dataset.Dataset(df.values.T, descriptors={'name': key})
        data_list.append(dataset)
        names.append(key)
    return data_list, names

def compute_rdms(data_list, **kwargs):
    """
    Compute RDMs for all datasets using error_correlation method.

    'correlation': Computes the Pearson correlation coefficient between patterns and converts it to a dissimilarity measure.
    'euclidean': Uses the Euclidean distance between patterns to quantify dissimilarity.
    'mahalanobis': Applies Mahalanobis distance, which accounts for correlations between variables.
    'crossnobis': A cross-validated version of the noise-normalized Mahalanobis distance.
    'poisson': Assumes Poisson-like noise characteristics for the data.
    'poisson_correlation': Similar to 'poisson', but uses correlation-based dissimilarity.
    """
    from rsatoolbox.rdm.calc import calc_rdm
    rdms = calc_rdm(data_list, **kwargs)
    return rdms

def analyze_rdm_similarity(rdms, method='spearman'):
    """
    Perform second-order RSA to compare the similarity between different RDMs.

    Parameters
    ----------
    rdms : rsatoolbox.rdm.RDMs
        The RDMs object containing dissimilarities and descriptors.
    method : str
        Correlation method: 'spearman' (default) or 'pearson'.

    Returns
    -------
    pd.DataFrame
        Table containing pairwise RDM similarities.
    """
    # 1. Extract data from the RDMs object
    # dissimilarities shape is (n_rdms, n_pairs), e.g., (9, 351)
    dist_vectors = rdms.dissimilarities
    names = rdms.rdm_descriptors.get("name", [f"RDM_{i}" for i in range(dist_vectors.shape[0])])
    n_rdms = dist_vectors.shape[0]

    rsa_results = []

    # 2. Iterate through unique pairs of RDMs
    for (idx1, name1), (idx2, name2) in combinations(enumerate(names), 2):
        vec1 = dist_vectors[idx1, :]
        vec2 = dist_vectors[idx2, :]

        # 3. Calculate correlation (Second-order RSA)
        if method == 'spearman':
            corr, _ = spearmanr(vec1, vec2)
        else:
            corr = np.corrcoef(vec1, vec2)[0, 1]

        rsa_results.append({
            'Group_A': name1,
            'Group_B': name2,
            'Pair': f"{name1} vs {name2}",
            'RSA': corr
        })

    # Create the final dataframe
    rsa_df = pd.DataFrame(rsa_results)

    # Sort by similarity descending to see the most similar studies first
    rsa_df = rsa_df.sort_values(by='RSA', ascending=False).reset_index(drop=True)

    return rsa_df

def plot_rsa_heatmap(rsa_df, ax=None, figsize=(6, 5), save_path=None):
    """
    Optional: Visualize the RSA similarity table as a heatmap.
    """

    # Pivot the long-format table back to a square matrix for the heatmap
    # We combine Group_A and Group_B to ensure all labels are present on both axes
    labels = sorted(list(set(rsa_df['Group_A']).union(set(rsa_df['Group_B']))))
    matrix = pd.DataFrame(1.0, index=labels, columns=labels)

    for _, row in rsa_df.iterrows():
        matrix.loc[row['Group_A'], row['Group_B']] = row['RSA']
        matrix.loc[row['Group_B'], row['Group_A']] = row['RSA']

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(matrix, annot=True, cmap='RdBu_r', vmin=-1, vmax=1, center=0, ax=ax)

    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight')

    if ax is None:  # Only show if we created the figure here
        plt.show()


# ### author_year

# In[54]:


data_list, names = extract_fitted_datasets(indices_by_subj, groups="author_year")
rdms = compute_rdms(data_list, method = "correlation")


# In[109]:


fig, ax, ret_val = rsatoolbox.vis.show_rdm(
    rdms,
    rdm_descriptor='name',
    # show_colorbar="figure",
    figsize=(7, 7)
)


# In[55]:


# This uses Spearman correlation by default
author_rsa_comparison_df = analyze_rdm_similarity(rdms)
mean_rsa_author = author_rsa_comparison_df.RSA.mean()
# author_rsa_comparison_df.to_csv("author_rdm_similarity.csv", index=False)


# ### task_name

# In[39]:


data_list, names = extract_fitted_datasets(indices_by_subj, groups="task_name")
rdms = compute_rdms(data_list, method = "correlation")


# In[112]:


fig, ax, ret_val = rsatoolbox.vis.show_rdm(
    rdms,
    rdm_descriptor='name',
    # show_colorbar="figure",
    figsize=(4, 4)
)


# In[ ]:


# This uses Spearman correlation by default
task_rsa_comparison_df = analyze_rdm_similarity(rdms)
mean_rsa_task = task_rsa_comparison_df.RSA.mean()
# task_rsa_comparison_df.to_csv("task_rdm_similarity.csv", index=False)


# ## Quantify weights difference between tasks and labs

# In[15]:


import arviz as az


# In[9]:


data_list, names = extract_fitted_datasets(indices_by_subj, groups="task_id")
rdms = compute_rdms(data_list, method = "correlation")


# In[10]:


# This uses Spearman correlation by default
id_rsa_comparison_df = analyze_rdm_similarity(rdms)
id_mean_rsa = id_rsa_comparison_df.RSA.mean()


# In[ ]:


def prepare_rsa_data(df):
    """
    Prepares the RSA dataframe for Bayesian analysis.

    1. Parses 'Group_A' and 'Group_B' to extract Author and Task.
       Expected format: "authorYEARtask" (e.g., "eisenberg2019flanker").
    2. Creates binary predictors: 'is_same_task' and 'is_same_author'.
    3. Applies Fisher Z-transform to the 'RSA' column to unbound the metric.

    Parameters
    ----------
    df : pd.DataFrame
        The output from analyze_rdm_similarity containing 'Group_A', 'Group_B', 'RSA'.

    Returns
    -------
    pd.DataFrame
        Augmented dataframe ready for Bambi.
    """
    data = df.copy()

    # 1. Parsing Helper Function
    def parse_name(name):
        # Regex: Matches (AuthorText)(4Digits)(TaskText)
        # e.g., "eisenberg2019flanker" -> ("eisenberg", "2019", "flanker")
        match = re.search(r'^([a-zA-Z]+)(\d{4})([a-zA-Z0-9_]+)$', name)
        if match:
            return match.group(1), match.group(3) # Return Author, Task
        return name, name # Fallback if format doesn't match

    # 2. Extract features
    # Apply parsing to both A and B columns
    data[['Author_A', 'Task_A']] = data['Group_A'].apply(lambda x: pd.Series(parse_name(x)))
    data[['Author_B', 'Task_B']] = data['Group_B'].apply(lambda x: pd.Series(parse_name(x)))

    # 3. Create Binary Predictors
    data['is_same_task'] = (data['Task_A'] == data['Task_B']).astype(int)
    data['is_same_author'] = (data['Author_A'] == data['Author_B']).astype(int)

    # 4. Fisher Z Transform
    # We clip to avoid infinity if perfect correlation (1.0) exists
    data['RSA_z'] = np.arctanh(data['RSA'].clip(lower=-0.999, upper=0.999))

    return data

df_ready = prepare_rsa_data(id_rsa_comparison_df)
# df_ready.to_csv("42_RSA_task_and_lab_difference.csv")
df_ready


# In[3]:


df_ready = pd.read_csv('42_RSA_task_and_lab_difference.csv')


# In[ ]:


def fit_rsa_model(df, 
                  formula="RSA_z ~ is_same_task + is_same_author + (1|Group_A) + (1|Group_B)",
                  save_name=None,
                  draws=2000, 
                  tune=1000, 
                  chains=4,
                  random_seed=42):
    """
    Fits a Bayesian Linear Mixed Model to RSA data using Bambi.

    Model:
        RSA_z ~ Normal(mu, sigma)
        mu = Intercept + beta_task * Task + beta_author * Author + RandomEffects

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe from `prepare_rsa_data`.
    formula : str
        Bambi formula string. We include random intercepts for Group_A and Group_B
        to account for the non-independence of dyadic data.
    save_name : str, optional
        Filename to load/save the InferenceData (NetCDF).

    Returns
    -------
    bmb.Model, az.InferenceData
    """

    import bambi as bmb
    import os
    import arviz as az

    # 1. Define Model
    print(f"Initializing model: {formula}")
    model = bmb.Model(formula, data=df, family="gaussian")

    file_path = f"{save_name}.nc" if save_name else None

    # 2. Check for cached results
    if file_path and os.path.exists(file_path):
        print(f"Loading existing results from {file_path}...")
        idata = az.from_netcdf(file_path)
    else:
        print("Fitting model (this may take a minute)...")
        # 3. Fit Model
        idata = model.fit(
            draws=draws, 
            tune=tune, 
            chains=chains, 
            random_seed=random_seed,
            idata_kwargs={"log_likelihood": True}
        )

        # 4. Save results
        if file_path:
            print(f"Saving results to {file_path}...")
            idata.to_netcdf(file_path)

    return model, idata

model, idata = fit_rsa_model(
    df_ready, 
    draws=4000, tune=2000, chains=4, # Low numbers for testing only
    save_name="42_RSA_task_and_lab_difference" 
)


# In[9]:


summary_df = az.summary(idata, var_names=['Intercept', 'is_same_task', 'is_same_author'], hdi_prob=0.95)


# ### plot and comparision

# In[30]:


sns.set_theme(style="white")
COLORS_DARKER = ['#75cfb6', '#4dc3eb']

def plot_effects_comparison(
    idata, 
    colors=COLORS_DARKER,
    save_path=None, 
    figsize=(4, 4),
    ax=None,
    tick_label_size=14
):
    """
    Visualize the posterior distribution comparison of Task vs Author effects using Seaborn.
    """

    # 1. Extract data (Extract Posterior Samples)
    post = idata.posterior
    # Assume bambi or pymc store variable names as follows
    b_task = post["is_same_task"].values.flatten()
    b_author = post["is_same_author"].values.flatten()

    # 2. Compute statistics (Compute Statistics)
    diff = b_author - b_task
    prob_task_greater = (diff > 0).mean()

    # Prepare DataFrame required for Seaborn (Long Format)
    df_plot = pd.DataFrame({
        "Same Tasks": b_task,
        "Same Labs": b_author
    }).melt(var_name="Condition", value_name="Effect Size")

    # 3. Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Use Seaborn Violinplot
    # inner="quart" displays quartile lines, perfect for showing posterior distributions
    sns.violinplot(
        data=df_plot,
        x="Condition",
        y="Effect Size",
        palette=colors,
        linewidth=1.5,
        inner="quart", 
        saturation=0.9,
        ax=ax
    )

    # 4. Annotation
    # Get maximum value of data to determine bracket height
    y_max = df_plot["Effect Size"].max()
    # Dynamically adjust height, leaving space
    y_range = df_plot["Effect Size"].max() - df_plot["Effect Size"].min()
    bracket_h = y_max + (y_range * 0.05)
    bar_tip = y_range * 0.02
    text_offset = y_range * 0.05

    # Draw brackets (assuming x-axis indices: 0 for Tasks, 1 for Labs)
    x1, x2 = 0, 1
    ax.plot(
        [x1, x1, x2, x2],
        [bracket_h, bracket_h + bar_tip, bracket_h + bar_tip, bracket_h],
        c="#333333",
        lw=1.5
    )

    # Add text description
    evidence = "Strong" if prob_task_greater > 0.95 else "Inconclusive"
    # If probability is very close to 1, display >0.999
    p_text = f"{prob_task_greater:.3f}" if prob_task_greater < 0.999 else ">0.999"

    label_text = (f"P(Lab > Task) = {p_text}\n"
                  f"({evidence} Evidence)")

    ax.text(
        (x1 + x2) * 0.5,
        bracket_h + text_offset,
        label_text,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="#333333"
    )

    # 5. Formatting
    ax.set_ylabel("Effect on RSA (Fisher Z)", fontsize=12, labelpad=10)
    ax.set_xlabel("") # Remove extra x-axis label

    # Add 0 reference line
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5, zorder=0)

    # Adjust Y-axis range to ensure text is not clipped
    ax.set_ylim(top=bracket_h + text_offset * 2.5)

    # Adjust tick label sizes
    ax.tick_params(axis='x', labelsize=tick_label_size)
    ax.tick_params(axis='y', labelsize=tick_label_size)

    # Beautify borders
    sns.despine(trim=True, offset=5)

    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight')

    return fig, ax

# plot_effects_comparison(idata)


# In[56]:


# plot_rsa_heatmap(author_rsa_comparison_df, save_path = "../figs/rsa_comparison_author_year.svg")
# plot_rsa_heatmap(task_rsa_comparison_df, save_path = "../figs/rsa_comparison_task_name.svg", figsize=(5,4))

author_rsa_comparison_df['Group_A'] = author_rsa_comparison_df['Group_A'].apply(format_author_year)
author_rsa_comparison_df['Group_B'] = author_rsa_comparison_df['Group_B'].apply(format_author_year)

task_rsa_comparison_df['Group_A'] = task_rsa_comparison_df['Group_A'].apply(format_task_name)
task_rsa_comparison_df['Group_B'] = task_rsa_comparison_df['Group_B'].apply(format_task_name)


# In[90]:


def plot_merged_rsa(author_df, task_df, idata=None, colors=COLORS_DARKER, figsize=(18, 5), save_path=None):
    """
    Merge Author and Task RSA heatmaps into one figure with labels A and B, and add effects comparison as C.
    """
    from matplotlib.gridspec import GridSpec

    # 1. Create a figure with custom grid layout to control spacing between subplots
    fig = plt.figure(figsize=figsize)
    # Use width_ratios to control relative widths and spacing
    # [1, 1, 0.2, 1] - the 0.2 creates a narrow space between the heatmaps and the effect plot
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 0.13, 0.8])

    ax1 = fig.add_subplot(gs[0, 0])  # Author RSA (A)
    ax2 = fig.add_subplot(gs[0, 1])  # Task RSA (B) 
    ax3 = fig.add_subplot(gs[0, 3])  # Effects Comparison (C)

    # 2. Plot Author RSA (Subplot A)
    plot_rsa_heatmap(author_df, ax=ax1)
    ax1.set_title("A", loc='left', x=-0.35, fontsize=14, fontweight='bold', pad=20)
    # ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha='right')

    # 3. Plot Task RSA (Subplot B)
    plot_rsa_heatmap(task_df, ax=ax2)
    ax2.set_title("B", loc='left', x=-0.2, fontsize=14, fontweight='bold', pad=20)

    # 4. Plot Effects Comparison (Subplot C)
    plot_effects_comparison(idata, colors=colors, ax=ax3)
    ax3.set_title("C", loc='left', x=-0.3, fontsize=14, fontweight='bold', pad=20)

    # 6. Remove redundant colorbars and add a single shared one for heatmaps only
    # Remove the individual colorbars created by sns.heatmap inside your function
    for ax in [ax1, ax2]:
        # Remove the individual colorbars created by sns.heatmap inside your function
        if ax.collections:
            try:
                cb = ax.collections[0].colorbar
                if cb:
                    cb.remove()
            except:
                pass

    # Create a shared colorbar axis for the heatmaps, positioned to the right of subplot B
    pos2 = ax2.get_position()
    # Position colorbar right after subplot B with some spacing
    cbar_ax = fig.add_axes([pos2.x1, pos2.y0, 0.010, pos2.height-0.1])
    sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=plt.Normalize(vmin=-1, vmax=1))
    fig.colorbar(sm, cax=cbar_ax, label='RSA Correlation (Spearman)')

    plt.setp(ax1.get_xticklabels(), rotation=20, ha='right')
    gs.figure.subplots_adjust(wspace=0.4)

    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        print(f"Merged plot saved to {save_path}")

plot_merged_rsa(
    author_rsa_comparison_df, 
    task_rsa_comparison_df, 
    idata=idata,  
    save_path="../figs/42_merged_rsa_comparison.svg"
)


# In[119]:


# output mean rsa for task and authors
print(f"mean rsa for task and authors {mean_rsa_task:.3f} vs. {mean_rsa_author:.3f}")


# In[ ]:




