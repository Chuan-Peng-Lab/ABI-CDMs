#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.metrics import r2_score
from scipy.stats import median_abs_deviation
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from utils import cache_from_file
from NSBI_CDMs import NSBICDM


# In[2]:


m_DDM = NSBICDM(model="DDM")
m_DMC = NSBICDM(model="DMC")
m_SSP = NSBICDM(model="SSP")
m_DSTP = NSBICDM(model="DSTP")

models = {
    "DDM": m_DDM,
    "SSP": m_SSP,
    "DMC": m_DMC,
    "DSTP": m_DSTP
}


# ## Loss History

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_single_loss(
    train_losses,
    val_losses=None,
    ax=None,
    moving_average=False,
    ma_window_fraction=0.01,
    train_color="#377eb8",
    val_color="black",
    lw_train=2,
    lw_val=3,
    grid_alpha=0.5,
    legend_fontsize=14,
    label_fontsize=14,
    title=None,
    title_fontsize=16,
    xlabel="Training step #",
    ylabel="Loss",
):
    """Plot a single loss curve with optional validation loss and moving average.

    Parameters
    ----------
    train_losses : array-like
        Training loss values for each step.
    val_losses : array-like or None, optional
        Validation loss values. If None, only training loss is plotted.
    ax : matplotlib.axes.Axes or None, optional
        Axes object to plot on. If None, creates a new figure.
    moving_average : bool, optional
        Whether to add a moving average line for training losses.
    ma_window_fraction : float, optional
        Window size for moving average as a fraction of total steps.
    train_color : str, optional
        Color for the training loss curve.
    val_color : str, optional
        Color for the validation loss curve.
    lw_train : int, optional
        Line width for the training loss curve.
    lw_val : int, optional
        Line width for the validation loss curve.
    grid_alpha : float, optional
        Opacity for the gridlines.
    legend_fontsize : int, optional
        Font size for the legend.
    label_fontsize : int, optional
        Font size for the axis labels.
    title : str or None, optional
        Title for the plot.
    title_fontsize : int, optional
        Font size for the title.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the plot.
    """
    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Prepare step indices
    train_losses = np.asarray(train_losses)
    train_step_index = np.arange(1, len(train_losses) + 1)

    # Plot training loss
    ax.plot(
        train_step_index,
        train_losses,
        color=train_color,
        lw=lw_train,
        alpha=0.9,
        label="Training",
    )

    # Add moving average for training loss
    if moving_average:
        window = max(1, int(len(train_losses) * ma_window_fraction))
        # Use numpy convolution for moving average
        cumsum = np.cumsum(np.insert(train_losses, 0, 0))
        smoothed = (cumsum[window:] - cumsum[:-window]) / window
        smoothed_index = train_step_index[window - 1 :]
        ax.plot(
            smoothed_index,
            smoothed,
            color="grey",
            lw=lw_train,
            label="Training (MA)",
        )

    # Plot validation loss if provided
    if val_losses is not None:
        val_losses = np.asarray(val_losses)
        val_step = max(1, len(train_losses) // len(val_losses))
        val_step_index = train_step_index[val_step - 1 :: val_step][: len(val_losses)]
        ax.plot(
            val_step_index,
            val_losses,
            linestyle="--",
            marker="o",
            color=val_color,
            lw=lw_val,
            label="Validation",
        )

    # Styling
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
    if title:
        ax.set_title(title, fontsize=title_fontsize)

    sns.despine(ax=ax)
    ax.grid(alpha=grid_alpha)

    # Show legend if validation or moving average is present
    if val_losses is not None or moving_average:
        ax.legend(fontsize=legend_fontsize)

    return ax


def plot_model_losses(
    models_dict,
    figsize=(10, 6),
    nrows=2,
    ncols=2,
    moving_average=True,
    ma_window_fraction=0.01,
    train_color="#377eb8",
    val_color="black",
    lw_train=2,
    lw_val=3,
    grid_alpha=0.5,
    legend_fontsize=14,
    label_fontsize=18,
    title_fontsize=22,
    save_path=None,
    dpi=600,
    ylabel_models=None,
    xlabel_models=None,
):
    """Plot training and validation losses for multiple models in a grid layout.

    Parameters
    ----------
    models_dict : dict
        Dictionary mapping model names to model objects.
        Each model must have a 'history' attribute with keys 'loss' and optionally 'val_loss'.
    figsize : tuple, optional
        Figure size (width, height) in inches.
    nrows : int, optional
        Number of rows in the subplot grid.
    ncols : int, optional
        Number of columns in the subplot grid.
    moving_average : bool, optional
        Whether to add moving average lines for training losses.
    ma_window_fraction : float, optional
        Window size for moving average as a fraction of total steps.
    train_color : str, optional
        Color for the training loss curves.
    val_color : str, optional
        Color for the validation loss curves.
    lw_train : int, optional
        Line width for training loss curves.
    lw_val : int, optional
        Line width for validation loss curves.
    grid_alpha : float, optional
        Opacity for gridlines.
    legend_fontsize : int, optional
        Font size for legends.
    label_fontsize : int, optional
        Font size for axis labels.
    title_fontsize : int, optional
        Font size for subplot titles.
    save_path : str or None, optional
        Path to save the figure. If None, figure is not saved.
    dpi : int, optional
        Resolution for saved figure.
    ylabel_models : list or None, optional
        List of model names that should display y-axis labels.
        If None, only leftmost column shows y-labels.
    xlabel_models : list or None, optional
        List of model names that should display x-axis labels.
        If None, only bottom row shows x-labels.

    Returns
    -------
    tuple
        (fig, axes) - The figure and axes array objects.

    Examples
    --------
    >>> fig, axes = plot_model_losses(
    ...     models_dict={'DDM': model1, 'DMC': model2, 'SSP': model3, 'LBA': model4},
    ...     ylabel_models=['DDM', 'DMC'],
    ...     xlabel_models=['DMC', 'LBA'],
    ...     save_path='train_loss.pdf'
    ... )
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = axes.flat if isinstance(axes, np.ndarray) else [axes]

    # Auto-determine which models should have labels if not specified
    model_names = list(models_dict.keys())
    if ylabel_models is None:
        # Show y-label for leftmost column
        ylabel_models = [
            model_names[i]
            for i in range(len(model_names))
            if i % ncols == 0
        ]
    if xlabel_models is None:
        # Show x-label for bottom row
        start_idx = max(0, len(model_names) - ncols)
        xlabel_models = model_names[start_idx:]

    # Plot each model
    for (model_name, model), ax in zip(models_dict.items(), axes_flat):
        history = model.history

        # Extract losses
        train_losses = history["loss"]
        val_losses = history.get("val_loss", None)

        # Determine labels
        ylabel = "Loss" if model_name in ylabel_models else ""
        xlabel = "Training step #" if model_name in xlabel_models else ""

        # Plot single loss curve
        plot_single_loss(
            train_losses=train_losses,
            val_losses=val_losses,
            ax=ax,
            moving_average=moving_average,
            ma_window_fraction=ma_window_fraction,
            train_color=train_color,
            val_color=val_color,
            lw_train=lw_train,
            lw_val=lw_val,
            grid_alpha=grid_alpha,
            legend_fontsize=legend_fontsize,
            label_fontsize=label_fontsize,
            title=model_name,
            title_fontsize=title_fontsize,
            xlabel=xlabel,
            ylabel=ylabel,
        )

    # Hide unused subplots
    for i in range(len(models_dict), len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()

    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, axes


fig, axes = plot_model_losses(
    models_dict=models,
    figsize=(10, 6),
    nrows=2,
    ncols=2,
    moving_average=True,
    train_color="#377eb8",
    ylabel_models=["DDM", "DMC"],
    xlabel_models=["DMC", "LBA"],
    # save_path="../figs/11_train_loss.svg",
    dpi=600
)


# ## Recovery

# In[32]:


def plot_recovery(
    post_samples,
    prior_samples,
    point_agg=np.median,
    uncertainty_agg=median_abs_deviation,
    param_names=None,
    fig_size=None,
    label_fontsize=16,
    title_fontsize=18,
    metric_fontsize=16,
    tick_fontsize=12,
    add_corr=True,
    add_r2=True,
    color="#8f2727",
    n_col=None,
    n_row=None,
    xlabel="Ground truth",
    ylabel="Estimated",
    axes=None,
    **kwargs,
):
    """Creates and plots publication-ready recovery plot with true vs. point estimate + uncertainty.
    The point estimate can be controlled with the ``point_agg`` argument, and the uncertainty estimate
    can be controlled with the ``uncertainty_agg`` argument.

    This plot yields similar information as the "posterior z-score", but allows for generic
    point and uncertainty estimates:

    https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html

    Important: Posterior aggregates play no special role in Bayesian inference and should only
    be used heuristically. For instance, in the case of multi-modal posteriors, common point
    estimates, such as mean, (geometric) median, or maximum a posteriori (MAP) mean nothing.

    Parameters
    ----------
    post_samples      : np.ndarray of shape (n_data_sets, n_post_draws, n_params)
        The posterior draws obtained from n_data_sets
    prior_samples     : np.ndarray of shape (n_data_sets, n_params)
        The prior draws (true parameters) obtained for generating the n_data_sets
    point_agg         : callable, optional, default: ``np.median``
        The function to apply to the posterior draws to get a point estimate for each marginal.
        The default computes the marginal median for each marginal posterior as a robust
        point estimate.
    uncertainty_agg   : callable or None, optional, default: scipy.stats.median_abs_deviation
        The function to apply to the posterior draws to get an uncertainty estimate.
        If ``None`` provided, a simple scatter using only ``point_agg`` will be plotted.
    param_names       : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
    fig_size          : tuple or None, optional, default : None
        The figure size passed to the matplotlib constructor. Inferred if None.
    label_fontsize    : int, optional, default: 16
        The font size of the y-label text
    title_fontsize    : int, optional, default: 18
        The font size of the title text
    metric_fontsize   : int, optional, default: 16
        The font size of the goodness-of-fit metric (if provided)
    tick_fontsize     : int, optional, default: 12
        The font size of the axis tick labels
    add_corr          : bool, optional, default: True
        A flag for adding correlation between true and estimates to the plot
    add_r2            : bool, optional, default: True
        A flag for adding R^2 between true and estimates to the plot
    color             : str, optional, default: '#8f2727'
        The color for the true vs. estimated scatter points and error bars
    n_row             : int, optional, default: None
        The number of rows for the subplots. Dynamically determined if None.
    n_col             : int, optional, default: None
        The number of columns for the subplots. Dynamically determined if None.
    xlabel            : str, optional, default: 'Ground truth'
        The label on the x-axis of the plot
    ylabel            : str, optional, default: 'Estimated'
        The label on the y-axis of the plot
    axes              : matplotlib.axes.Axes or np.ndarray of Axes, optional, default: None
        The axes to plot on. If provided, `n_col` and `n_row` are ignored.
    **kwargs          : optional
        Additional keyword arguments passed to ax.errorbar or ax.scatter.
        Example: `rasterized=True` to reduce PDF file size with many dots

    Returns
    -------
    ax : matplotlib.axes.Axes or np.ndarray of Axes - the axes instance for optional further modifications

    Raises
    ------
    ShapeError
        If there is a deviation from the expected shapes of ``post_samples`` and ``prior_samples``.
    """

    # Compute point estimates and uncertainties
    est = point_agg(post_samples, axis=1)
    if uncertainty_agg is not None:
        u = uncertainty_agg(post_samples, axis=1)

    # Determine n params and param names if None given
    n_params = prior_samples.shape[-1]
    if param_names is None:
        param_names = [f"$\\theta_{{{i}}}$" for i in range(1, n_params + 1)]

    # If axes are provided, use them directly
    if axes is not None:
        axarr = np.array(axes).flatten()
        n_row, n_col = np.array(axes).shape if np.array(axes).ndim > 1 else (1, len(axes))
    else:
        # Determine number of rows and columns for subplots based on inputs
        if n_row is None and n_col is None:
            n_row = int(np.ceil(n_params / 6))
            n_col = int(np.ceil(n_params / n_row))
        elif n_row is None and n_col is not None:
            n_row = int(np.ceil(n_params / n_col))
        elif n_row is not None and n_col is None:
            n_col = int(np.ceil(n_params / n_row))

        # Initialize figure
        if fig_size is None:
            fig_size = (int(4 * n_col), int(4 * n_row))
        f, axarr = plt.subplots(n_row, n_col, figsize=fig_size)
        axarr = axarr.flat

    for i, ax in enumerate(axarr):
        if i >= n_params:
            break

        # Add scatter and error bars
        if uncertainty_agg is not None:
            _ = ax.errorbar(prior_samples[:, i], est[:, i], yerr=u[:, i], fmt="o", alpha=0.5, color=color, **kwargs)
        else:
            _ = ax.scatter(prior_samples[:, i], est[:, i], alpha=0.5, color=color, **kwargs)

        # Make plots quadratic to avoid visual illusions
        lower = min(prior_samples[:, i].min(), est[:, i].min())
        upper = max(prior_samples[:, i].max(), est[:, i].max())
        eps = (upper - lower) * 0.1
        ax.set_xlim([lower - eps, upper + eps])
        ax.set_ylim([lower - eps, upper + eps])
        ax.plot(
            [ax.get_xlim()[0], ax.get_xlim()[1]],
            [ax.get_ylim()[0], ax.get_ylim()[1]],
            color="black",
            alpha=0.9,
            linestyle="dashed",
        )

        # Add optional metrics and title
        if add_r2:
            r2 = r2_score(prior_samples[:, i], est[:, i])
            ax.text(
                0.1,
                0.9,
                "$R^2$ = {:.3f}".format(r2),
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
                size=metric_fontsize,
            )
        if add_corr:
            corr = np.corrcoef(prior_samples[:, i], est[:, i])[0, 1]
            ax.text(
                0.1,
                0.8,
                "$r$ = {:.3f}".format(corr),
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
                size=metric_fontsize,
            )
        ax.set_title(param_names[i], fontsize=title_fontsize)

        # Prettify
        sns.despine(ax=ax)
        ax.grid(alpha=0.5)
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        ax.tick_params(axis="both", which="minor", labelsize=tick_fontsize)

    # Only add x-labels to the bottom row
    bottom_row = axarr if n_row == 1 else axarr[-n_col:]
    for _ax in bottom_row:
        _ax.set_xlabel(xlabel, fontsize=label_fontsize)

    # Only add y-labels to the left-most column
    if n_row == 1:  # if there is only one row, the ax array is 1D
        axarr[0].set_ylabel(ylabel, fontsize=label_fontsize)
    else:  # If there is more than one row, the ax array is 2D
        for _ax in axarr[::n_col]:
            _ax.set_ylabel(ylabel, fontsize=label_fontsize)

    # Remove unused axes entirely
    for _ax in axarr[n_params:]:
        _ax.remove()

    if axes is None:
        f.tight_layout()
        return f
    else:
        return axes


# In[33]:


@cache_from_file("11parameter_recovery_results.pkl")
def generate_test_simulations_and_posterior_samples(models, n_obs=100, n_pos=1000):
    """
    Generate test simulations and posterior samples for each model.

    Parameters:
        models (dict): Dictionary of models where keys are model names and values are model objects.
        n_obs (int): Number of prior sets (default: 100).
        n_pos (int): Number of posterior samples (default: 1000).

    Returns:
        tuple: A tuple containing:
            - test_sims (dict): Test simulations for each model.
            - posterior_samples (dict): Posterior samples for each model.
    """

    # Iterate over models and estimate parameters
    test_sims = {}
    posterior_samples = {}

    for m_name, m_i in models.items():
        with torch.no_grad():
            # Generate test simulations and posterior samples
            test_sims_i = m_i.workflow.simulate(n_obs)
            posterior_samples_i = m_i.workflow.sample(conditions=test_sims_i, num_samples=n_pos)

        # print_gpu_memory()

        test_sims[m_name] = test_sims_i
        posterior_samples[m_name] = posterior_samples_i

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return test_sims, posterior_samples


test_sims, posterior_samples = generate_test_simulations_and_posterior_samples(models)


# In[34]:


def plot_grid(
    models,
    posterior_samples,
    test_sims,
    plot_func,
    n_rows=4,
    n_cols=7,
    figsize_multiplier=4,
    label_width_ratio=0.2,
    hide_patterns=None,
    model_label_fontsize=40,
    save_path=None,
    dpi=600,
    **plot_kwargs
):
    """
    Create a grid of plots for multiple models.

    Parameters
    ----------
    models : dict
        Dictionary with model names as keys and model objects as values.
        Each model object must have:
        - param_keys: list of parameter keys (for data extraction)
        - param_names: list of parameter display names

    posterior_samples : dict
        Dictionary with model names as keys and nested dictionaries as values.
        Each nested dictionary contains parameter keys and corresponding posterior samples.

    test_sims : dict
        Dictionary with model names as keys and nested dictionaries as values.
        Each nested dictionary contains parameter keys and corresponding prior/true samples.

    plot_func : callable
        Function to plot individual parameter recovery plots.
        Must accept: post_samples, prior_samples, param_names, axes

    n_rows : int, default=4
        Number of rows in the grid (typically equals number of models).

    n_cols : int, default=7
        Number of parameter columns (excluding the label column).

    figsize_multiplier : int or float, default=4
        Base size multiplier for figure dimensions.

    label_width_ratio : float, default=0.2
        Ratio of label column width to plot column width (2:10 = 0.2).

    hide_patterns : list of tuples, optional
        Patterns of subplots to hide. Each tuple should be (row, column_start, condition).
        If None, uses default patterns: 
        [(0, 5), (1, 6), (2, 7)] for rows 0, 1, 2 respectively.

    label_fontsize : int, default=40
        Font size for model name labels in the leftmost column.

    save_path : str, optional
        Path to save the figure. If None, figure is not saved.

    dpi : int, default=600
        Resolution for saved figure.

    **plot_kwargs
        Additional keyword arguments passed to plot_func function.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.

    axs : list of list of matplotlib.axes.Axes
        2D list of axes objects in the grid.
    """

    # Calculate figure size based on number of columns and rows
    fig_width = figsize_multiplier * (n_cols + 1)  # +1 for label column
    fig_height = figsize_multiplier * n_rows
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Create GridSpec with width ratio for label column
    label_width = int(label_width_ratio * 10)  # Convert ratio to GridSpec units
    plot_width = 10
    gs = GridSpec(
        n_rows, 
        n_cols + 1,  # +1 for label column
        width_ratios=[label_width] + [plot_width] * n_cols,
        figure=fig
    )

    # Create subplots grid
    axs = []
    for i in range(n_rows):
        row_axs = []
        for j in range(n_cols + 1):  # +1 for label column
            if j == 0:
                # Add subplot for model label (leftmost column)
                ax = fig.add_subplot(gs[i, j])
                ax.axis("off")  # Hide axes for label area
                row_axs.append(ax)
            else:
                # Add subplot for parameter recovery plot
                ax = fig.add_subplot(gs[i, j])
                row_axs.append(ax)
        axs.append(row_axs)

    # Set default hide patterns if not provided
    if hide_patterns is None:
        hide_patterns = [
            (0, 5),  # Row 0: hide columns 5 and beyond
            (1, 6),  # Row 1: hide columns 6 and beyond
            (2, 7),  # Row 2: hide column 7
        ]

    # Hide specified subplots
    for row, start_col in hide_patterns:
        if row < n_rows:  # Check if row exists
            for j in range(start_col, n_cols + 1):
                if j <= n_cols:  # Check if column exists
                    axs[row][j].set_visible(False)

    # Iterate over models and create recovery plots
    for idx, (model_name, model_obj) in enumerate(models.items()):
        if idx >= n_rows:
            break  # Safety check: don't exceed allocated rows

        # Get parameter keys and names for current model
        param_keys = model_obj.param_keys
        param_names = model_obj.param_names

        # Extract posterior samples for current model
        posterior_samples_i = np.stack(
            [posterior_samples[model_name][key] for key in param_keys],
            axis=-1
        ).squeeze()

        # Extract prior/true samples for current model
        prior_samples_i = np.stack(
            [test_sims[model_name][key] for key in param_keys],
            axis=-1
        ).squeeze()

        # Get axes for current row (excluding label column)
        plot_axes = axs[idx][1:]  # Skip first column (label)

        # Plot recovery for current model
        plot_func(
            post_samples=posterior_samples_i,
            prior_samples=prior_samples_i,
            param_names=param_names,
            axes=plot_axes,
            **plot_kwargs
        )

        if idx < n_rows - 1:
            for ax in plot_axes:
                ax.set_xlabel('')

        # Add model name label in leftmost column
        axs[idx][0].text(
            0.5, 0.5,  # Center of the axis
            model_name,
            fontsize=model_label_fontsize,
            ha="center",  # Horizontal alignment
            va="center",  # Vertical alignment
            rotation=90,  # Vertical text orientation
            transform=axs[idx][0].transAxes  # Use axis coordinates
        )

    # Adjust layout
    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, axs


# In[35]:


fig, axs = plot_grid(
    models=models,
    posterior_samples=posterior_samples,
    test_sims=test_sims,
    plot_func=plot_recovery,
    n_rows=4,
    n_cols=7,
    figsize_multiplier=4,
    model_label_fontsize=40,
    save_path="../figs/11_parameters_recovery.svg",
    dpi=600,
    # Additional kwargs for plot_recovery
    title_fontsize=32,
    label_fontsize=23,
    tick_fontsize=20,
    color="#377eb8"
)


# ## plot ecdf and z-score

# In[40]:


from collections.abc import Callable, Mapping, Sequence

from bayesflow.utils.dict_utils import compute_test_quantities
from bayesflow.utils.plot_utils import prepare_plot_data, add_titles_and_labels, prettify_subplots
from bayesflow.utils.ecdf import simultaneous_ecdf_bands
from bayesflow.utils.ecdf.ranks import fractional_ranks, distance_ranks
from bayesflow.utils.plot_utils import prepare_plot_data, add_titles_and_labels, prettify_subplots

def calibration_ecdf(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray,
    variable_keys: Sequence[str] = None,
    variable_names: Sequence[str] = None,
    test_quantities: dict[str, Callable] = None,
    difference: bool = False,
    stacked: bool = False,
    rank_type: str | np.ndarray = "fractional",
    figsize: Sequence[float] = None,
    axes: np.ndarray | plt.Axes | list = None,  # Updated type hint
    label_fontsize: int = 16,
    legend_fontsize: int = 14,
    legend_location: str = "upper right",
    title_fontsize: int = 18,
    tick_fontsize: int = 12,
    rank_ecdf_color: str = "#132a70",
    fill_color: str = "grey",
    num_row: int = None,
    num_col: int = None,
    **kwargs,
) -> plt.Figure:
    """
    (Docstring omitted for brevity - same as before)
    """

    # Optionally, compute and prepend test quantities from draws
    if test_quantities is not None:
        updated_data = compute_test_quantities(
            targets=targets,
            estimates=estimates,
            variable_keys=variable_keys,
            variable_names=variable_names,
            test_quantities=test_quantities,
        )
        variable_names = updated_data["variable_names"]
        variable_keys = updated_data["variable_keys"]
        estimates = updated_data["estimates"]
        targets = updated_data["targets"]

    # 1. Standard path: Let utils create figure
    if axes is None:
        plot_data = prepare_plot_data(
            estimates=estimates,
            targets=targets,
            variable_keys=variable_keys,
            variable_names=variable_names,
            num_col=num_col,
            num_row=num_row,
            figsize=figsize,
            stacked=stacked,
        )

    # 2. Custom axes path: Manually prepare data
    else:
        # --- ROBUST DATA MERGING LOGIC ---
        # Handle Dictionary input for estimates
        if isinstance(estimates, dict):
            if variable_keys is None:
                variable_keys = list(estimates.keys())

            # Robust merge: handle both (N, M) and (N, M, 1) shapes
            est_list = []
            for k in variable_keys:
                data = estimates[k]
                if data.ndim == 2: # (N, M) -> (N, M, 1)
                    data = data[..., np.newaxis]
                est_list.append(data)
            # Use concatenate along last axis to get (N, M, D)
            estimates = np.concatenate(est_list, axis=-1)

            # Infer variable names if not provided
            if variable_names is None:
                variable_names = variable_keys

        # Handle Dictionary input for targets
        if isinstance(targets, dict):
             if variable_keys is None: 
                  variable_keys = list(targets.keys())

             # Robust merge for targets: (N,) or (N, 1)
             tgt_list = []
             for k in variable_keys:
                data = targets[k]
                if data.ndim == 1: # (N,) -> (N, 1)
                    data = data[..., np.newaxis]
                tgt_list.append(data)
             targets = np.concatenate(tgt_list, axis=-1)

        # Basic defaults
        if variable_names is None:
            variable_names = [f"Parameter {i+1}" for i in range(estimates.shape[-1])]

        # --- AXES HANDLING ---
        # Convert list of axes (or single axis) to numpy array
        if isinstance(axes, list):
            axes_array = np.array(axes).flatten() # Flatten list to 1D array
        elif not isinstance(axes, np.ndarray):
            axes_array = np.array([axes])
        else:
            axes_array = axes.flatten() # Ensure flat for iteration

        # Validation
        if stacked:
            if axes_array.size < 1:
                raise ValueError("When stacked=True, at least 1 axis must be provided.")
        else:
            if axes_array.size < estimates.shape[-1]:
                raise ValueError(f"Provided axes ({axes_array.size}) are fewer than parameters ({estimates.shape[-1]}).")

        # Construct plot_data
        # Infer rows and cols from the provided axes array shape if not provided
        if num_row is None:
            if axes_array.ndim > 1:
                num_row = axes_array.shape[0]
            else:
                num_row = 1

        if num_col is None:
            if axes_array.ndim > 1:
                num_col = axes_array.shape[1]
            else:
                num_col = axes_array.shape[0]

        plot_data = {
            "estimates": estimates,
            "targets": targets,
            "variable_names": variable_names,
            "axes": axes_array,
            # Retrieve figure from the first axis
            "fig": axes_array.flat[0].figure if axes_array.size > 0 else None,
            "num_variables": estimates.shape[-1],
            "num_row": num_row,
            "num_col": num_col
        }

    estimates = plot_data.pop("estimates")
    targets = plot_data.pop("targets")

    if rank_type == "fractional":
        ranks = fractional_ranks(estimates, targets)
    elif rank_type == "distance":
        ranks = distance_ranks(estimates, targets, stacked=stacked, **kwargs.pop("ranks_kwargs", {}))
    else:
        raise ValueError(f"Unknown rank type: {rank_type}. Use 'fractional' or 'distance'.")

    # Plot individual ecdf of parameters
    # Note: ranks.shape is (N, D). Loop iterates D times.
    for j in range(ranks.shape[-1]):
        xx = np.repeat(np.sort(ranks[:, j]), 2)
        xx = np.pad(xx, (1, 1), constant_values=(0, 1))
        yy = np.linspace(0, 1, num=xx.shape[-1] // 2)
        yy = np.repeat(yy, 2)

        if difference:
            yy -= xx

        if stacked:
            # Always plot on the first axis if stacked
            ax = plot_data["axes"].flat[0]
            if j == 0:
                ax.plot(xx, yy, color=rank_ecdf_color, alpha=0.95, label="Rank ECDFs")
            else:
                ax.plot(xx, yy, color=rank_ecdf_color, alpha=0.95)
        else:
            # Plot on j-th axis
            plot_data["axes"].flat[j].plot(xx, yy, color=rank_ecdf_color, alpha=0.95, label="Rank ECDF")

    # Compute uniform ECDF and bands
    alpha, z, L, U = simultaneous_ecdf_bands(estimates.shape[0], **kwargs.pop("ecdf_bands_kwargs", {}))

    if difference:
        L -= z
        U -= z
        ylab = "ECDF Difference"
    else:
        ylab = "ECDF"

    # Add simultaneous bounds
    if not stacked:
        titles = plot_data["variable_names"]
        # Use zip to pair up parameters with axes safely
        axes_to_use = plot_data["axes"].flat
    elif rank_type in ["distance", "random"]:
        titles = ["Joint ECDFs"]
        axes_to_use = [plot_data["axes"].flat[0]]
    else:
        titles = ["Stacked ECDFs"]
        axes_to_use = [plot_data["axes"].flat[0]]

    for ax, title in zip(axes_to_use, titles):
        ax.fill_between(z, L, U, color=fill_color, alpha=0.2, label=rf"{int((1 - alpha) * 100)}$\%$ Confidence Bands")
        ax.legend(fontsize=legend_fontsize, loc=legend_location)
        ax.set_title(title, fontsize=title_fontsize)

    prettify_subplots(plot_data["axes"], num_subplots=plot_data["num_variables"], tick_fontsize=tick_fontsize)

    add_titles_and_labels(
        plot_data["axes"],
        plot_data["num_row"],
        plot_data["num_col"],
        xlabel=f"{rank_type.capitalize()} rank statistic",
        ylabel=ylab,
        label_fontsize=label_fontsize,
    )

    # Only layout if we own the figure
    if axes is None:
        plot_data["fig"].tight_layout()

    return plot_data["fig"]

def z_score_contraction(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray,
    variable_keys: Sequence[str] = None,
    variable_names: Sequence[str] = None,
    figsize: Sequence[int] = None,
    axes: np.ndarray | plt.Axes | list = None,
    label_fontsize: int = 16,
    title_fontsize: int = 18,
    tick_fontsize: int = 12,
    color: str = "#132a70",
    num_col: int = None,
    num_row: int = None,
    markersize: float = None,
) -> plt.Figure:
    """
    Implements a graphical check for global model sensitivity by plotting the
    posterior z-score over the posterior contraction for each set of posterior
    samples in ``estimates`` according to [1].

    - The definition of the posterior z-score is:

    post_z_score = (posterior_mean - true_parameters) / posterior_std

    And the score is adequate if it centers around zero and spreads roughly
    in the interval [-3, 3]

    - The definition of posterior contraction is:

    post_contraction = 1 - (posterior_variance / prior_variance)

    In other words, the posterior contraction is a proxy for the reduction in
    uncertainty gained by replacing the prior with the posterior.
    The ideal posterior contraction tends to 1.
    Contraction near zero indicates that the posterior variance is almost
    identical to the prior variance for the particular marginal parameter
    distribution.

    Note:
    Means and variances will be estimated via their sample-based estimators.

    [1] Schad, D. J., Betancourt, M., & Vasishth, S. (2021).
    Toward a principled Bayesian workflow in cognitive science.
    Psychological methods, 26(1), 103.

    Paper also available at https://arxiv.org/abs/1904.12765

    Parameters
    ----------
    estimates       : np.ndarray of shape (num_datasets, num_post_draws, num_params)
        The posterior draws obtained from num_datasets
    targets         : np.ndarray of shape (num_datasets, num_params)
        The prior draws (true parameters) used for generating the num_datasets
    variable_keys       : list or None, optional, default: None
       Select keys from the dictionaries provided in estimates and targets.
       By default, select all keys.
    variable_names    : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
    figsize           : tuple or None, optional, default : None
        The figure size passed to the matplotlib constructor. Inferred if None.
    axes              : np.ndarray or plt.Axes, optional, default: None
        Custom matplotlib axes to plot on.
        If provided, `figsize`, `num_row`, and `num_col` are ignored (unless needed for layout logic).
    label_fontsize    : int, optional, default: 16
        The font size of the y-label text
    title_fontsize    : int, optional, default: 18
        The font size of the title text
    tick_fontsize     : int, optional, default: 12
        The font size of the axis ticklabels
    color             : str, optional, default: '#8f2727'
        The color for the true vs. estimated scatter points and error bars
    num_row           : int, optional, default: None
        The number of rows for the subplots. Dynamically determined if None.
    num_col           : int, optional, default: None
        The number of columns for the subplots. Dynamically determined if None.
    markersize        : float, optional, default: None
        The marker size in points**2 of the scatter plot.

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    Raises
    ------
    ShapeError
        If there is a deviation from the expected shapes of ``estimates`` and ``targets``.
    """

    # 1. Standard path: Create new figure and axes using utils
    if axes is None:
        plot_data = prepare_plot_data(
            estimates=estimates,
            targets=targets,
            variable_keys=variable_keys,
            variable_names=variable_names,
            num_col=num_col,
            num_row=num_row,
            figsize=figsize,
        )

    # 2. Custom axes path: Manually prepare data
    else:
        # --- ROBUST DATA MERGING LOGIC ---
        # Handle Dictionary input for estimates
        if isinstance(estimates, dict):
            if variable_keys is None:
                variable_keys = list(estimates.keys())

            # Robust merge: handle both (N, M) and (N, M, 1) shapes
            est_list = []
            for k in variable_keys:
                data = estimates[k]
                if data.ndim == 2: # (N, M) -> (N, M, 1)
                    data = data[..., np.newaxis]
                est_list.append(data)
            # Use concatenate along last axis to get (N, M, D)
            estimates = np.concatenate(est_list, axis=-1)

            # Infer variable names if not provided
            if variable_names is None:
                variable_names = variable_keys

        # Handle Dictionary input for targets
        if isinstance(targets, dict):
             if variable_keys is None: 
                  variable_keys = list(targets.keys())

             # Robust merge for targets: (N,) or (N, 1)
             tgt_list = []
             for k in variable_keys:
                data = targets[k]
                if data.ndim == 1: # (N,) -> (N, 1)
                    data = data[..., np.newaxis]
                tgt_list.append(data)
             targets = np.concatenate(tgt_list, axis=-1)

        # Basic defaults
        if variable_names is None:
            variable_names = [f"Parameter {i+1}" for i in range(estimates.shape[-1])]

        # --- AXES HANDLING ---
        # Convert list of axes (or single axis) to numpy array
        if isinstance(axes, list):
            axes_array = np.array(axes).flatten() # Flatten list to 1D array
        elif not isinstance(axes, np.ndarray):
            axes_array = np.array([axes])
        else:
            axes_array = axes.flatten()

        # Validation
        if axes_array.size < estimates.shape[-1]:
            raise ValueError(f"Provided axes ({axes_array.size}) are fewer than parameters ({estimates.shape[-1]}).")

        # Infer rows and cols from the provided axes array shape if not provided
        # This is critical for add_titles_and_labels to work correctly
        if num_row is None:
            # Note: Since we flattened above for safety, we might lose 2D structure.
            # However, if the user passed a list or 1D array, 
            # we simply assume 1 row or calculate loosely.
            # Ideally, we look at the original input shape if it was an ndarray.
            if isinstance(axes, np.ndarray) and axes.ndim > 1:
                num_row = axes.shape[0]
            else:
                num_row = 1 

        if num_col is None:
            if isinstance(axes, np.ndarray) and axes.ndim > 1:
                num_col = axes.shape[1]
            else:
                 num_col = axes_array.size

        plot_data = {
            "estimates": estimates,
            "targets": targets,
            "variable_names": variable_names,
            "axes": axes_array,
            # Retrieve figure from the first axis
            "fig": axes_array[0].figure if axes_array.size > 0 else None,
            "num_variables": estimates.shape[-1],
            "num_row": num_row,
            "num_col": num_col
        }

    estimates = plot_data.pop("estimates")
    targets = plot_data.pop("targets")

    # Estimate posterior means and stds
    post_means = estimates.mean(axis=1)
    post_vars = estimates.var(axis=1, ddof=1)
    post_stds = np.sqrt(post_vars)

    # Estimate prior variance
    # Note: This assumes 'targets' contains samples drawn from the prior
    prior_vars = targets.var(axis=0, keepdims=True, ddof=1)

    # Compute contraction and z-score
    # Avoid division by zero if prior variance is 0 (e.g. fixed parameter)
    with np.errstate(divide='ignore', invalid='ignore'):
        contraction = 1 - (post_vars / prior_vars)
        # Clip only lower bound 0, upper bound can theoretically be slightly > 1 due to estimator noise, 
        # but logically should be <= 1. Standard implementation clips both.
        contraction = np.clip(contraction, 0, 1)
        z_score = (post_means - targets) / post_stds

    # Loop and plot
    for i, ax in enumerate(plot_data["axes"].flat):
        if i >= plot_data["num_variables"]:
            break

        ax.scatter(contraction[:, i], z_score[:, i], color=color, alpha=0.5, s=markersize)
        ax.set_xlim([-0.05, 1.05])

    prettify_subplots(plot_data["axes"], num_subplots=plot_data["num_variables"], tick_fontsize=tick_fontsize)

    # Add labels, titles, and set font sizes
    add_titles_and_labels(
        axes=plot_data["axes"],
        num_row=plot_data["num_row"],
        num_col=plot_data["num_col"],
        title=plot_data["variable_names"],
        xlabel="Posterior contraction",
        ylabel="Posterior z-score",
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
    )

    # Only layout if we own the figure
    if axes is None:
        plot_data["fig"].tight_layout()

    return plot_data["fig"]


# In[37]:


def plot_grid(
    models,
    posterior_samples,
    test_sims,
    plot_func,
    n_rows=4,
    n_cols=7,
    figsize_multiplier=4,
    label_width_ratio=0.2,
    hide_patterns=None,
    model_label_fontsize=40,
    save_path=None,
    dpi=600,
    **plot_kwargs
):
    """
    Create a grid of plots for multiple models.

    Parameters
    ----------
    models : dict
        Dictionary with model names as keys and model objects as values.
        Each model object must have:
        - param_keys: list of parameter keys (for data extraction)
        - param_names: list of parameter display names

    posterior_samples : dict
        Dictionary with model names as keys and nested dictionaries as values.
        Each nested dictionary contains parameter keys and corresponding posterior samples.

    test_sims : dict
        Dictionary with model names as keys and nested dictionaries as values.
        Each nested dictionary contains parameter keys and corresponding prior/true samples.

    plot_func : callable
        Function to plot individual parameter recovery plots.
        Must accept: post_samples, prior_samples, param_names, axes

    n_rows : int, default=4
        Number of rows in the grid (typically equals number of models).

    n_cols : int, default=7
        Number of parameter columns (excluding the label column).

    figsize_multiplier : int or float, default=4
        Base size multiplier for figure dimensions.

    label_width_ratio : float, default=0.2
        Ratio of label column width to plot column width (2:10 = 0.2).

    hide_patterns : list of tuples, optional
        Patterns of subplots to hide. Each tuple should be (row, column_start, condition).
        If None, uses default patterns: 
        [(0, 5), (1, 6), (2, 7)] for rows 0, 1, 2 respectively.

    label_fontsize : int, default=40
        Font size for model name labels in the leftmost column.

    save_path : str, optional
        Path to save the figure. If None, figure is not saved.

    dpi : int, default=600
        Resolution for saved figure.

    **plot_kwargs
        Additional keyword arguments passed to plot_func function.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.

    axs : list of list of matplotlib.axes.Axes
        2D list of axes objects in the grid.
    """

    # Calculate figure size based on number of columns and rows
    fig_width = figsize_multiplier * (n_cols + 1)  # +1 for label column
    fig_height = figsize_multiplier * n_rows
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Create GridSpec with width ratio for label column
    label_width = int(label_width_ratio * 10)  # Convert ratio to GridSpec units
    plot_width = 10
    gs = GridSpec(
        n_rows, 
        n_cols + 1,  # +1 for label column
        width_ratios=[label_width] + [plot_width] * n_cols,
        figure=fig
    )

    # Create subplots grid
    axs = []
    for i in range(n_rows):
        row_axs = []
        for j in range(n_cols + 1):  # +1 for label column
            if j == 0:
                # Add subplot for model label (leftmost column)
                ax = fig.add_subplot(gs[i, j])
                ax.axis("off")  # Hide axes for label area
                row_axs.append(ax)
            else:
                # Add subplot for parameter recovery plot
                ax = fig.add_subplot(gs[i, j])
                row_axs.append(ax)
        axs.append(row_axs)

    # Set default hide patterns if not provided
    if hide_patterns is None:
        hide_patterns = [
            (0, 5),  # Row 0: hide columns 5 and beyond
            (1, 6),  # Row 1: hide columns 6 and beyond
            (2, 7),  # Row 2: hide column 7
        ]

    # Hide specified subplots
    for row, start_col in hide_patterns:
        if row < n_rows:  # Check if row exists
            for j in range(start_col, n_cols + 1):
                if j <= n_cols:  # Check if column exists
                    axs[row][j].set_visible(False)

    # Iterate over models and create recovery plots
    for idx, (model_name, model_obj) in enumerate(models.items()):
        if idx >= n_rows:
            break  # Safety check: don't exceed allocated rows

        # Extract posterior samples and simulated data for current model
        estimates_i = posterior_samples[model_name]
        targets_i = test_sims[model_name]

        # Get axes for current row (excluding label column)
        plot_axes = axs[idx][1:]  # Skip first column (label)

        # Plot recovery for current model
        plot_func(
            estimates=estimates_i,
            targets=targets_i,
            variable_keys = model_obj.param_keys,
            variable_names = model_obj.param_names,
            axes=plot_axes,
            **plot_kwargs
        )

        if idx < n_rows - 1:
            for ax in plot_axes:
                ax.set_xlabel('')

        # Add model name label in leftmost column
        axs[idx][0].text(
            0.5, 0.5,  # Center of the axis
            model_name,
            fontsize=model_label_fontsize,
            ha="center",  # Horizontal alignment
            va="center",  # Vertical alignment
            rotation=90,  # Vertical text orientation
            transform=axs[idx][0].transAxes  # Use axis coordinates
        )

    # Adjust layout
    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, axs


# In[38]:


fig, axs = plot_grid(
    models=models,
    posterior_samples=posterior_samples,
    test_sims=test_sims,
    plot_func=calibration_ecdf,
    n_rows=4,
    n_cols=7,
    figsize_multiplier=4,
    model_label_fontsize=40,
    save_path="../figs/11_ecdf.svg",
    dpi=600,
    # Additional kwargs for plot_recovery
    title_fontsize=32,
    label_fontsize=23,
    tick_fontsize=20,
    rank_ecdf_color="#377eb8"
)


# In[41]:


fig, axs = plot_grid(
    models=models,
    posterior_samples=posterior_samples,
    test_sims=test_sims,
    plot_func=z_score_contraction,
    n_rows=4,
    n_cols=7,
    figsize_multiplier=4,
    model_label_fontsize=40,
    save_path="../figs/11_z_score.svg",
    dpi=600,
    # Additional kwargs for plot_recovery
    title_fontsize=32,
    label_fontsize=23,
    tick_fontsize=20,
    color="#377eb8"
)

