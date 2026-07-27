##===========================================================================
#                            functions extracted from pydmc: https://github.com/igmmgi/pydmc
#===========================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.mstats import mquantiles
from fastkde import fastKDE
from typing import Union
import inspect


__all__ = [
    "Ob",
    "Plot",
    "PlotFit",
    "calculate_rmse_cost",
    "calculate_chi_square_cost",
]

# Safe aggregation functions
def safe_nanmean(data):
    
    if isinstance(data, pd.Series):
        if data.dropna().empty:
            return np.nan
        return np.nanmean(data)
    elif isinstance(data, np.ndarray):
        if np.isnan(data).all():
            return np.nan
        return np.nanmean(data)
    else:
        raise TypeError("Input must be a pd.Series or np.ndarray")


def safe_nanstd(data, ddof=1):
    # Check the type of the input data
    if isinstance(data, pd.Series):
        # If it's a pd.Series, check if the number of non-NaN elements is greater than ddof
        if len(data.dropna()) <= ddof:
            return np.nan
        return np.nanstd(data, ddof=ddof)
    elif isinstance(data, np.ndarray):
        # If it's a np.ndarray, check if the number of non-NaN elements is greater than ddof
        if np.count_nonzero(~np.isnan(data)) <= ddof:
            return np.nan
        return np.nanstd(data, ddof=ddof)
    else:
        raise TypeError("Input must be a pd.Series or np.ndarray")



def aggregate_trials(
    data: pd.DataFrame,
    groups=["Subject", "Comp"],
    df_col=["Error", "RT", "outlier"]
) -> None:
    def aggfun(x: pd.DataFrame):

        x = x[df_col].to_numpy()
        ind_error = x[:, 0] == 1
        ind_correct = x[:, 0] == 0
        # outlier marked as 1
        ind_outlier = x[:, 2] == 1

        dat_agg = pd.DataFrame(
            {
                "n": [x.shape[0]],
                "n_cor": [np.sum(ind_correct)],
                "n_err": [np.sum(ind_error)],
                "n_out": [np.sum(ind_outlier)],
                "rt_cor": [safe_nanmean(x[ind_correct & ~ind_outlier, 1])],
                "rt_err": [safe_nanmean(x[ind_error & ~ind_outlier, 1])]
            }
        )
        dat_agg["per_err"] = (
            dat_agg["n_err"] / (dat_agg["n_err"] + dat_agg["n_cor"]) * 100
        )

        return dat_agg

    data = data.dropna()
    out = (data.groupby(groups).apply(aggfun, include_groups=False).reset_index()).drop("level_2", axis=1)

    return out


def calculate_caf(data: pd.DataFrame, n_caf: int = 5) -> None:
    """Calculate conditional accuracy functions."""
    def caffun(x, n: int) -> pd.DataFrame:
        # remove outliers and bin data
        # x = x[x.outlier == 0].reset_index()
        cafbin = np.digitize(
            x.loc[:, "RT"],
            np.percentile(x.loc[:, "RT"], np.linspace(0, 100, n + 1)),
        )
        x = x.assign(bin=cafbin)
        return pd.DataFrame((1 - x.groupby(["bin"])["Error"].mean())[:-1])

    caf_subject = (
        data
        .groupby(["Subject", "Comp"])
        .apply(caffun, n_caf, include_groups=False)
        .reset_index()
        .pivot(
            index=("Subject", "bin"), columns="Comp", values="Error")
        .reset_index()
        .rename_axis(None, axis=1)
        .assign(
            effect=lambda x: (x["comp"] - x["incomp"]) * 100
        )
    )

    caf = (caf_subject.groupby("bin").mean().reset_index().drop("Subject", axis=1))

    return (caf_subject, caf)


class Ob:
    def __init__(
        self,
        data: pd.DataFrame,
        n_caf: int = 5,
        n_delta: int = 19,
        p_delta: tuple = (),
        t_delta: int = 1,
        outlier: tuple = (100, 3000),
        subj_col:str = "subject_id",
        rt_col:str = "rt",
        response_col:str = "accuracy",
        congruency_col:str = "congruency"
    ):

        self.data = df_to_pydmc(
            data.copy(), 
            subj_col = subj_col,
            rt_col = rt_col,
            response_col=response_col,
            congruency_col=congruency_col
        )
        self.columns = ("Subject", "Comp", "RT", "Error")
        assert all(
            [i in self.data.columns for i in self.columns]
        ), f"dataframe should contain the columns {self.columns}"
        
        assert self.data["RT"].max() > 10, "RT should be in ms"
        self.data = self.data[list(self.columns)]
        
        self.comp_coding = ("comp", "incomp")
        assert set(self.data["Comp"].unique() )== set(self.comp_coding), f"Value in column Comp must be {self.comp_coding}"
        self.error_coding = (0, 1)
        assert all(i in set(self.error_coding) for i in self.data["Error"].unique()), f"Value in column Error must be {self.error_coding}"


        self.outlier = outlier
        # Handle NaN values before applying outlier detection using .loc to avoid SettingWithCopyWarning
        self.data.loc[:, "RT"] = self.data["RT"].fillna(np.nan)
        # Identify outliers
        self.data.loc[:, "outlier"] = np.where(
            (self.data["RT"] <= self.outlier[0]) |
            (self.data["RT"] >= self.outlier[1]),
            1,
            0,
        )

        self.n_caf = n_caf
        self.n_delta = n_delta
        self.p_delta = p_delta
        self.t_delta = t_delta

        self.summary_subject = aggregate_trials(self.data)
        self.summary = self._aggregate_subjects(self.summary_subject)
        self.caf_subject, self.caf = calculate_caf(self.data, self.n_caf)
        self.delta_subject, self.delta = self._calc_delta_values()

    def _aggregate_subjects(self, summary_subject) -> None:
        def aggfun(x):
            n_subjects = len(x["Subject"])
            if n_subjects == 1:
                new_cols = [
                    [
                        n_subjects,
                        safe_nanmean(x["rt_cor"]),
                        np.nan,
                        np.nan,
                        safe_nanmean(x["rt_err"]),
                        np.nan,
                        np.nan,
                        safe_nanmean(x["per_err"]),
                        np.nan,
                        np.nan,
                    ]
                ]
            else:
                new_cols = [
                    [
                        n_subjects,
                        safe_nanmean(x["rt_cor"]),
                        safe_nanstd(x["rt_cor"], ddof=1),
                        safe_nanstd(x["rt_cor"], ddof=1) /
                        np.sqrt(x["Subject"].count()),
                        safe_nanmean(x["rt_err"]),
                        safe_nanstd(x["rt_err"], ddof=1),
                        safe_nanstd(x["rt_err"], ddof=1) /
                        np.sqrt(x["Subject"].count()),
                        safe_nanmean(x["per_err"]),
                        safe_nanstd(x["per_err"], ddof=1),
                        safe_nanstd(x["per_err"], ddof=1) /
                        np.sqrt(x["Subject"].count()),
                    ]
                ]

            dat_agg = pd.DataFrame(
                new_cols,
                columns=[
                    "n",
                    "rt_cor",
                    "sd_rt_cor",
                    "se_rt_cor",
                    "rt_err",
                    "sd_rt_err",
                    "se_rt_err",
                    "per_err",
                    "sd_per_err",
                    "se_per_err",
                ],
            )
            return dat_agg

        out = (summary_subject.groupby(
            ["Comp"]
        ).apply(aggfun, include_groups=False).reset_index()).drop("level_1", axis=1)

        return out

    def _calc_delta_values(self) -> None:
        """Calculate compatibility effect + delta values for correct trials."""

        # noinspection PyUnboundLocalVariable
        def deltafun(x: pd.DataFrame) -> pd.DataFrame:
            # filter trials
            x = x[(x.outlier == 0) & (x.Error == 0)].reset_index()

            if self.t_delta == 1:
                if len(self.p_delta) != 0:
                    percentiles = self.p_delta
                else:
                    percentiles = np.linspace(0, 1, self.n_delta + 2)[1:-1]

                mean_bins = np.array(
                    [
                        mquantiles(
                            x["RT"][(x["Comp"] == comp)],
                            percentiles,
                            alphap=0.5,
                            betap=0.5,
                        ) for comp in ("comp", "incomp")
                    ]
                )

            elif self.t_delta == 2:
                if len(self.p_delta) != 0:
                    percentiles = (0, ) + self.p_delta + (1, )
                else:
                    percentiles = np.linspace(0, 1, self.n_delta + 1)

                mean_bins = np.zeros((2, len(percentiles) - 1))
                for idx, comp in enumerate(("comp", "incomp")):
                    dat = x["RT"][(x["Comp"] == comp)]
                    bin_values = mquantiles(
                        dat,
                        percentiles,
                        alphap=0.5,
                        betap=0.5,
                    )
                    tile = np.digitize(dat, bin_values)
                    mean_bins[idx, :] = np.array(
                        [dat[tile == i].mean() for i in range(1, len(bin_values))]
                    )

            mean_bin = mean_bins.mean(axis=0)
            mean_effect = mean_bins[1, :] - mean_bins[0, :]

            dat = np.array(
                [
                    range(1,
                          len(mean_bin) + 1),
                    mean_bins[0, :],
                    mean_bins[1, :],
                    mean_bin,
                    mean_effect,
                ]
            ).T

            return pd.DataFrame(
                dat,
                columns=[
                    "bin", "mean_comp", "mean_incomp", "mean_bin", "mean_effect"
                ],
            )

        delta_subject = (
            self.data.groupby(["Subject"]).apply(deltafun, include_groups=False).reset_index()
        ).drop("level_1", axis=1)

        def aggfun(x: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    [
                        np.nanmean(x["mean_comp"]),
                        np.nanmean(x["mean_incomp"]),
                        np.nanmean(x["mean_bin"]),
                        np.nanmean(x["mean_effect"]),
                    ]
                ],
                columns=["mean_comp", "mean_incomp", "mean_bin", "mean_effect"],
            )

        delta = (
            delta_subject.groupby(["bin"]).apply(aggfun, include_groups=False).reset_index()
        ).drop("level_1", axis=1)

        return (delta_subject, delta)

    def select_subject(self, subject: int) -> pd.DataFrame:
        """Select subject"""
        return self.data[self.data.Subject == subject]
        
def calculate_rmse_cost(
    res_th: Union[Ob, pd.DataFrame], 
    res_ob: Union[Ob, pd.DataFrame],
    **kwargs
    ) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE) cost between two objects (either of type `Ob` or `pd.DataFrame`).
    The function computes the RMSE for two components: `caf` and `delta`, using data from the `res_th` (theoretical) and `res_ob` (observed) objects. 
    These components are combined with specific weights to generate a final RMSE cost.

    The RMSE cost is computed as follows:
        - For the `caf` (compensation and incompletion), the cost is computed based on the squared differences of the corresponding values in `res_th` and `res_ob`, normalized by the number of error terms (`n_err`).
        - For the `delta` (mean compensation and incompletion), the cost is computed similarly, normalized by the number of RT terms (`n_rt`).
        - A weighted average of the two costs is returned, with the `delta` component weighted more heavily based on the total number of RT terms (`n_rt`) and the `caf` component scaled by a factor of 1500.

    Parameters:
    ----------
    res_th : Union[Ob, pd.DataFrame]
        Theoretical data, either as an `Ob` object or a pandas DataFrame containing `caf` and `delta` attributes.

    res_ob : Union[Ob, pd.DataFrame]
        Observed data, either as an `Ob` object or a pandas DataFrame containing `caf` and `delta` attributes.

    Returns:
    -------
    float
        The weighted RMSE cost combining the `caf` and `delta` components.

    Notes:
    ------
    - The `Ob` object is expected to have `caf` and `delta` attributes, where `caf` contains the "comp" and "incomp" columns,
      and `delta` contains the "mean_comp" and "mean_incomp" columns.
    - The function first checks if the inputs are pandas DataFrames, and converts them to `Ob` objects if necessary.
    - The number of error terms (`n_err`) is derived from the length of `caf`, and the number of RT terms (`n_rt`) is derived from the length of `delta`.
    - The final cost is a weighted sum of the individual RMSEs for the `caf` and `delta` components.

    Example:
    --------
    cost = calculate_rmse_cost(res_th, res_ob)
    """

    if isinstance(res_th, pd.DataFrame):
        res_th = Ob(res_th, **kwargs)
    if isinstance(res_ob, pd.DataFrame):
        res_ob = Ob(res_ob, **kwargs)

    n_rt = len(res_th.delta) * 2
    n_err = len(res_th.caf) * 2

    cost_caf = np.sqrt(
        (1 / n_err)
        * np.sum(
            np.sum(
                (res_th.caf[["comp", "incomp"]] - res_ob.caf[["comp", "incomp"]])
                ** 2,
                axis=0
            )
        )
    )
    cost_rt = np.sqrt(
        (1 / n_rt)
        * np.sum(
            np.sum(
                (
                    res_th.delta[["mean_comp", "mean_incomp"]]
                    - res_ob.delta[["mean_comp", "mean_incomp"]]
                )
                ** 2,
                axis=0
            )
        )
    )

    weight_rt = n_rt / (n_rt + n_err)
    weight_caf = (1 - weight_rt) * 1500

    return (weight_caf * cost_caf) + (weight_rt * cost_rt)

def calculate_cost_calue_spe(res_th: Ob, res_ob: Ob) -> float:
    """calculate_cost_calue_spe (squared percentage error) or likelihood-ratio chi-square statistic

    Calculate Squared Percentage Error between simulated
    and observed data points

    Parameters
    ---------
    res_th
    res_ob
    """
    cost_caf = np.sum(
        ((res_ob.caf.iloc[:, 1:3] - res_th.caf.iloc[:, 1:3]) / res_ob.caf.iloc[:, 1:3]) ** 2,
        axis=0 
    ).sum()

    cost_rt = np.sum(
        ((res_ob.delta.iloc[:, 1:4] - res_th.delta.iloc[:, 1:4]) / res_ob.delta.iloc[:, 1:4]) ** 2,
        axis=0 
    ).sum()

    return cost_rt + cost_caf

def _filter_dict(given, allowed):
    """Internal function to filter dict **kwargs for allowd arguments."""
    f = {}
    allowed = inspect.signature(allowed).parameters
    for k, v in given.items():
        if k in allowed.keys():
            f[k] = v
    return f


def _adjust_plt(ax, **kwargs):
    """Internal function to adjust some common plot properties for a single axis."""
    if kwargs.get("xlim", None) is not None:
        ax.set_xlim(kwargs["xlim"])
    if kwargs.get("ylim", None) is not None:
        ax.set_ylim(kwargs["ylim"])
    ax.set_xlabel(
        kwargs.get("xlabel", ""), fontsize=kwargs.get("label_fontsize", 12)
    )
    ax.set_ylabel(
        kwargs.get("ylabel", ""), fontsize=kwargs.get("label_fontsize", 12)
    )
    ax.tick_params(axis='x', labelsize=kwargs.get("tick_fontsize", 10))
    ax.tick_params(axis='y', labelsize=kwargs.get("tick_fontsize", 10))

    return ax


def _plot_beh(ax, dat, cond_labels: tuple, zeroed: bool, **kwargs):
    """Internal function to plot rt/er for comp vs. comp on a single axis."""
    kwargs.setdefault("color", "black")
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("markersize", 4)

    l_kws = _filter_dict(kwargs, plt.Line2D)
    ax.plot(cond_labels, dat, **l_kws)
    ax.margins(x=0.5)

    if zeroed:
        kwargs.setdefault("ylim", [0, np.max(dat) + 5])
    else:
        kwargs.setdefault("ylim", [np.min(dat) - 100, np.max(dat) + 100])

    return _adjust_plt(ax, **kwargs)


class Plot:
    def __init__(self, res: Union[Ob, pd.DataFrame], **kwargs):

        if isinstance(res, pd.DataFrame):
            res = Ob(res, **kwargs)
        
        self.res = res

    def summary(self, **kwargs):
        """Plot summary."""
        kwargs.setdefault("fig_type", "summary1")
        kwargs.setdefault("hspace", 0.5)
        kwargs.setdefault("wspace", 0.5)

        plt.figure(len(plt.get_fignums()) + 1)

        # if isinstance(self.res, Sim):
        #     if kwargs["fig_type"] == "summary1" and not self.res.full_data:
        #         kwargs["fig_type"] = "summary2"
        #     if kwargs["fig_type"] == "summary1":
        #         self._summary1_res_th(**kwargs)
        #     elif kwargs["fig_type"] == "summary2":
        #         self._summary2_res_th(**kwargs)
        #     elif kwargs["fig_type"] == "summary3":
        #         self._summary3_res_th(**kwargs)
        # else:
        #     self._summary1_res_ob(**kwargs)
        self._summary1_res_ob(**kwargs)

    def _summary1_res_th(self, **kwargs):
        # upper left panel (activation)
        plt.subplot2grid((6, 4), (0, 0), rowspan=3, colspan=2)
        self.activation(show=False, **kwargs)

        # lower left panel (trials)
        plt.subplot2grid((6, 4), (3, 0), rowspan=3, colspan=2)
        self.trials(show=False, **kwargs)

        # upper right (left) panel (PDF)
        plt.subplot2grid((6, 4), (0, 2), rowspan=2)
        self.pdf(show=False, **kwargs)

        # upper right (right) panel (CDF)
        plt.subplot2grid((6, 4), (0, 3), rowspan=2)
        self.cdf(show=False, **kwargs)

        # middle right panel (CAF)
        plt.subplot2grid((6, 4), (2, 2), rowspan=2, colspan=2)
        self.caf(show=False, **kwargs)

        # bottom right panel (delta)
        plt.subplot2grid((6, 4), (4, 2), rowspan=2, colspan=2)
        self.delta(show=False, **kwargs)

        plt.subplots_adjust(hspace=kwargs["hspace"], wspace=kwargs["wspace"])
        plt.show(block=False)

    def _summary2_res_th(self, **kwargs):
        # upper right (left) panel (PDF)
        plt.subplot2grid((3, 2), (0, 0))
        self.pdf(show=False, **kwargs)

        # upper right (eight) panel (CDF)
        plt.subplot2grid((3, 2), (0, 1))
        self.cdf(show=False, **kwargs)

        # middle left panel
        plt.subplot2grid((3, 2), (1, 0), colspan=2)
        self.caf(show=False, **kwargs)

        # bottom right panel
        plt.subplot2grid((3, 2), (2, 0), colspan=2)
        self.delta(show=False, **kwargs)

        plt.subplots_adjust(hspace=kwargs["hspace"], wspace=kwargs["wspace"])
        plt.show(block=False)

    def _summary3_res_th(self, **kwargs):
        # upper right (left) panel (PDF)
        plt.subplot2grid((3, 1), (0, 0))
        self.rt_correct(show=False, **kwargs)

        # upper right (eight) panel (CDF)
        plt.subplot2grid((3, 1), (1, 0))
        self.er(show=False, **kwargs)

        # middle left panel
        plt.subplot2grid((3, 1), (2, 0))
        self.rt_error(show=False, **kwargs)

        plt.subplots_adjust(hspace=kwargs["hspace"], wspace=kwargs["wspace"])
        plt.show(block=False)

    def _summary1_res_ob(self, figsize=(10, 8), show=True, **kwargs):
        """Plot summaty observed data."""
        kwargs.setdefault("fig_type", "summary1")
        kwargs.setdefault("hspace", 0.5)
        kwargs.setdefault("wspace", 0.5)

        axs = kwargs.pop("axs", None)
        if axs is None:
            _, axs = plt.subplots(3, 2, figsize=figsize)
        else:
            assert len(axs.flatten()) == 6, "axs must have length of 6"

        # upper left panel (rt correct)
        axs[0, 0] = self.rt_correct(ax=axs[0, 0], show=False, **kwargs)
        # middle left panel
        axs[1, 0] = self.er(ax=axs[1, 0], show=False, **kwargs)
        # bottom left panel
        axs[2, 0] = self.rt_error(ax=axs[2, 0], show=False, **kwargs)
        # upper right panel (cdf)
        axs[0, 1] = self.cdf(ax=axs[0, 1], show=False, **kwargs)
        # middle right (left) panel (PDF)
        axs[1, 1] = self.caf(ax=axs[1, 1], show=False, **kwargs)
        # lower right (right) panel (CDF)
        axs[2, 1] = self.delta(ax=axs[2, 1], show=False, **kwargs)

        if show:
            plt.subplots_adjust(hspace=kwargs["hspace"], wspace=kwargs["wspace"])
            plt.show(block=False)
        else:
            return axs

    def activation(
        self,
        ax = None, 
        show: bool = True,
        cond_labels: tuple = ("Compatible", "Incompatible"),
        legend_position: str = "best",
        colors: tuple = ("green", "red"),
        **kwargs,
    ):
        """Plot activation."""

        # if not isinstance(self.res, Sim):
        #     print("Observed data does not have activation function!")
        #     return
        if ax is None:
            fig, ax = plt.subplots()

        if not self.res.xt:
            print("Plotting activation function requires full_data=True")
            return

        if show:
            plt.figure(len(plt.get_fignums()) + 1)

        l_kws = _filter_dict(kwargs, plt.Line2D)

        ax.plot(self.res.eq4, "k-")
        ax.plot(self.res.eq4 * -1, "k--")
        ax.plot(self.res.xt[0], color=colors[0], label=cond_labels[0], **l_kws)
        ax.plot(self.res.xt[1], color=colors[1], label=cond_labels[1], **l_kws)
        ax.plot(
            np.cumsum(np.repeat(self.res.prms.drc, self.res.prms.t_max)),
            color="black",
            **l_kws,
        )
        self._bounds()

        kwargs.setdefault("xlim", [0, self.res.prms.t_max])
        kwargs.setdefault(
            "ylim", [-self.res.prms.bnds - 20, self.res.prms.bnds + 20]
        )
        kwargs.setdefault("xlabel", "Time (ms)")
        kwargs.setdefault("ylabel", "E[X(t)]")

        _adjust_plt(ax,**kwargs)

        if legend_position is not None:
            ax.legend(loc=legend_position)

        if show:
            plt.show(block=False)

    def trials(
        self,
        ax=None, 
        show: bool = True,
        cond_labels: tuple = ("Compatible", "Incompatible"),
        legend_position: str = "upper right",
        colors: tuple = ("green", "red"),
        **kwargs,
    ):
        """Plot individual trials."""

        # if not isinstance(self.res, Sim):
        #     print("Observed data does not have individual trials!")
        #     return
        if ax is None:
            fig, ax = plt.subplots()

        if not self.res.xt:
            print("Plotting individual trials function requires full_data=True")
            return

        if show:
            plt.figure(len(plt.get_fignums()) + 1)

        l_kws = _filter_dict(kwargs, plt.Line2D)
        for trl in range(self.res.n_trls_data):
            if trl == 0:
                labels = cond_labels
            else:
                labels = [None, None]
            for comp in (0, 1):
                idx = np.where(
                    np.abs(self.res.data_trials[comp][trl, :]) >= self.res.prms.bnds
                )[0][0]
                plt.plot(
                    self.res.data_trials[comp][trl][0:idx],
                    color=colors[comp],
                    label=labels[comp],
                    **l_kws,
                )
        self._bounds()

        kwargs.setdefault("xlim", [0, self.res.prms.t_max])
        kwargs.setdefault(
            "ylim", [-self.res.prms.bnds - 20, self.res.prms.bnds + 20]
        )
        kwargs.setdefault("xlabel", "Time (ms)")
        kwargs.setdefault("ylabel", "X(t)")

        _adjust_plt(ax,**kwargs)

        if legend_position:
            ax.legend(loc=legend_position)

        if show:
            plt.show(block=False)

    def _bounds(self):
        plt.axhline(y=self.res.prms.bnds, color="grey", linestyle="--")
        plt.axhline(y=-self.res.prms.bnds, color="grey", linestyle="--")

    def pdf(
        self,
        ax=None,
        show: bool = True,
        cond_labels: tuple = ("Compatible", "Incompatible"),
        legend_position: str = "upper right",
        colors: tuple = ("green", "red"),
        **kwargs,
    ):
        """Plot PDF."""

        if ax is None:
            fig, ax = plt.subplots()

        l_kws = _filter_dict(kwargs, plt.Line2D)

        # if isinstance(self.res, Sim):
        #     for comp in (0, 1):
        #         pdf, axes = fastKDE.pdf(self.res.data[comp][0])
        #         plt.plot(
        #             axes, pdf, color=colors[comp], label=cond_labels[comp], **l_kws
        #         )
        #     kwargs.setdefault("xlim", [0, self.res.prms.t_max])
        if isinstance(self.res, Ob):
            for idx, comp in enumerate(("comp", "incomp")):
                data = np.array(self.res.data["RT"][self.res.data.Comp == comp])
                pdf, axes = fastKDE.pdf(data)
                ax.plot(
                    axes, pdf, color=colors[idx], label=cond_labels[idx], **l_kws
                )
            kwargs.setdefault("xlim", [min(data) - 100, max(data) + 100])

        kwargs.setdefault("ylim", [0, 0.01])
        kwargs.setdefault("xlabel", "Time (ms)")
        kwargs.setdefault("ylabel", "PDF")

        _adjust_plt(ax,**kwargs)

        if legend_position:
            ax.legend(loc=legend_position)

        if show:
            plt.show(block=False)

    def cdf(
        self,
        ax=None,
        show: bool = True,
        cond_labels: tuple = ("Compatible", "Incompatible"),
        legend_position: tuple = "lower right",
        colors: tuple = ("green", "red"),
        **kwargs,
    ):
        """Plot CDF."""

        if ax is None:
            fig, ax = plt.subplots()

        l_kws = _filter_dict(kwargs, plt.Line2D)
        if hasattr(self.res, "prms"):
            for comp in (0, 1):
                pdf, axes = fastKDE.pdf(self.res.data[comp][0])
                ax.plot(
                    axes,
                    np.cumsum(pdf) * np.diff(axes)[0:1],
                    color=colors[comp],
                    label=cond_labels[comp],
                    **l_kws,
                )

            kwargs.setdefault("xlim", [0, self.res.prms.t_max])

        else:
            kwargs.setdefault("marker", "o")
            kwargs.setdefault("markersize", 4)

            l_kws = _filter_dict(kwargs, plt.Line2D)
            for idx, comp in enumerate(("mean_comp", "mean_incomp")):
                ax.plot(
                    self.res.delta[comp],
                    np.linspace(0, 1, self.res.n_delta + 2)[1:-1],
                    color=colors[idx],
                    label=cond_labels[idx],
                    **l_kws,
                )

            kwargs.setdefault(
                "xlim",
                [
                    np.min(self.res.delta.mean_bin) - 100,
                    np.max(self.res.delta.mean_bin) + 100,
                ],
            )

        kwargs.setdefault("ylim", [0, 1.05])
        kwargs.setdefault("xlabel", "Time (ms)")
        kwargs.setdefault("ylabel", "CDF")

        ax.axhline(y=1, color="grey", linestyle="--")
        _adjust_plt(ax,**kwargs)

        if legend_position:
            ax.legend(loc=legend_position)

        if show:
            plt.show(block=False)

    def caf(
        self,
        ax=None,
        show: bool = True,
        cond_labels: tuple = ("Compatible", "Incompatible"),
        legend_position: str = "lower right",
        colors: tuple = ("green", "red"),
        **kwargs,
    ):
        """Plot CAF."""

        if ax is None:
            fig, ax = plt.subplots()

        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)

        l_kws = _filter_dict(kwargs, plt.Line2D)
        for idx, comp in enumerate(("comp", "incomp")):
            ax.plot(
                self.res.caf["bin"],
                self.res.caf[comp],
                color=colors[idx],
                label=cond_labels[idx],
                **l_kws,
            )

        ax.set_xticks(
            range(1, self.res.n_caf + 1),
            [str(x) for x in range(1, self.res.n_caf + 1)]
        )

        kwargs.setdefault("ylim", [0, 1.1])
        kwargs.setdefault("xlabel", "RT Bin")
        kwargs.setdefault("ylabel", "CAF")

        _adjust_plt(ax,**kwargs)

        if legend_position:
            ax.legend(loc=legend_position)

        if show:
            plt.show(block=False)

    def delta(
        self,
        ax = None,
        show: bool = True, 
        **kwargs):
        """Plot reaction-time delta plots."""

        if ax is None:
            fig, ax = plt.subplots()

        kwargs.setdefault("color", "black")
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)

        datx, daty = self.res.delta["mean_bin"], self.res.delta["mean_effect"]

        l_kws = _filter_dict(kwargs, plt.Line2D)
        ax.plot(datx, daty, **l_kws)

        kwargs.setdefault("xlim", [np.min(datx) - 100, np.max(datx) + 100])
        kwargs.setdefault("ylim", [np.min(daty) - 25, np.max(daty) + 25])
        kwargs.setdefault("xlabel", "Time (ms)")
        kwargs.setdefault("ylabel", r"$\Delta$  RT [ms]")

        _adjust_plt(ax,**kwargs)

        if show:
            plt.show(block=False)

    def delta_errors(
        self, 
        ax = None,
        show: bool = True, 
        **kwargs):
        """Plot error rate delta plots."""

        if ax is None:
            fig, ax = plt.subplots()

        kwargs.setdefault("color", "black")
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)

        datx, daty = self.res.caf["bin"], self.res.caf["effect"]

        l_kws = _filter_dict(kwargs, plt.Line2D)
        ax.plot(datx, daty, **l_kws)

        ax.set_xticks(
            range(1, self.res.n_caf + 1),
            [str(x) for x in range(1, self.res.n_caf + 1)]
        )

        kwargs.setdefault("ylim", [np.min(daty) - 5, np.max(daty) + 5])
        kwargs.setdefault("xlabel", "RT Bin")
        kwargs.setdefault("ylabel", r"$\Delta$  ER [%]")

        _adjust_plt(ax,**kwargs)

        if show:
            plt.show(block=False)

    def rt_correct(
        self,
        ax=None,
        show: bool = True,
        cond_labels: tuple = ("Compatible", "Incompatible"),
        **kwargs,
    ):
        """Plot correct RT's."""

        if ax is None:
            fig, ax = plt.subplots()

        kwargs.setdefault("ylabel", "RT Correct [ms]")

        _plot_beh(
            ax,
            self.res.summary["rt_cor"],
            cond_labels,
            False,
            **kwargs,
        )
        
        if show:
            plt.show(block=False)

    def er(
        self,
        ax=None,
        show: bool = True,
        cond_labels: tuple = ("Compatible", "Incompatible"),
        **kwargs,
    ):
        """Plot error rate"""

        if ax is None:
            fig, ax = plt.subplots()

        kwargs.setdefault("ylabel", "Error Rate [%]")

        _plot_beh(
            ax,
            self.res.summary["per_err"],
            cond_labels,
            False,
            **kwargs,
        )
        
        if show:
            plt.show(block=False)

    def rt_error(
        self,
        ax=None,
        show: bool = True,
        cond_labels: tuple = ("Compatible", "Incompatible"),
        **kwargs,
    ):
        """Plot error RT's."""

        if ax is None:
            fig, ax = plt.subplots()

        kwargs.setdefault("ylabel", "RT Error [ms]")

        _plot_beh(
            ax,
            self.res.summary["rt_err"],
            cond_labels,
            False,
            **kwargs,
        )

        if show:
            plt.show(block=False)

def df_to_pydmc(
    df:pd.DataFrame, 
    subj_col = "subject_id", 
    rt_col = "rt", 
    response_col = "accuracy", 
    congruency_col = "congruency"
    ):
    df = df.rename(
        columns={
            subj_col: "Subject",
            rt_col: "RT",
            response_col: "Error",
            congruency_col: "Comp"
        }
    )

    df["RT"] = abs(df["RT"]) * 1000  # milisecond to second
    df["Error"] = np.where(df["Error"] == 1, 0, 1)
    df["Comp"] = np.where(df["Comp"] == 1, "comp", "incomp")

    return df

class PlotFit:
    def __init__(
        self,
        obs_data: pd.DataFrame,
        sim_data: pd.DataFrame,
        **kwargs
    ):
        # assert isinstance(res, Fit), "res must be of type 'Fit'" # NOTE

        self.res_ob = Ob(obs_data, **kwargs)
        self.res_th = Ob(sim_data, **kwargs)

    def summary(self, figsize=(10, 8), show=True, **kwargs):
        """Plot."""
        kwargs.setdefault("fig_type", "summary1")
        kwargs.setdefault("hspace", 0.5)
        kwargs.setdefault("wspace", 0.5)

        axs = kwargs.pop("axs", None)
        if axs is None:
            _, axs = plt.subplots(3, 2, figsize=figsize)
        else:
            assert len(axs.flatten()) == 6, "axs must have length of 6"

        axs[0, 0] = self.rt_correct(ax=axs[0, 0], show=False, **kwargs)
        axs[1, 0] = self.er(ax=axs[1, 0], show=False, **kwargs)
        axs[2, 0] = self.rt_error(ax=axs[2, 0], show=False, **kwargs)
        axs[0, 1] = self.cdf(ax=axs[0, 1], show=False, **kwargs)
        axs[1, 1] = self.caf(ax=axs[1, 1], show=False, **kwargs)
        axs[2, 1] = self.delta(ax=axs[2, 1], show=False, **kwargs)

        if show:
            plt.subplots_adjust(hspace=kwargs["hspace"], wspace=kwargs["wspace"])
            plt.show(block=False)
        else:
            return axs

    def rt_correct(
        self,
        ax=None,
        show: bool = True,
        cond_labels: tuple = ("Compatible", "Incompatible"),
        legend_labels: tuple = ("Observed", "Predicted"),
        legend_position: str = "upper left",
        **kwargs,
    ):
        """Plot correct RT's."""
        if ax is None:
            fig, ax = plt.subplots()

        kwargs.setdefault("ylabel", "RT Correct [ms]")
        kwargs.setdefault("colors", ("black", "grey"))
        kwargs.setdefault("linestyles", ("-", "--"))
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)

        l_kws = _filter_dict(kwargs, plt.Line2D)
        for idx, res in enumerate((self.res_ob, self.res_th)):
            ax.plot(
                cond_labels,
                res.summary["rt_cor"],
                color=kwargs["colors"][idx],
                linestyle=kwargs["linestyles"][idx],
                label=legend_labels[idx],
                **l_kws,
            )

        kwargs.setdefault(
            "ylim",
            [
                np.min(self.res_ob.summary["rt_cor"]) - 100,
                np.max(self.res_ob.summary["rt_cor"]) + 100,
            ],
        )

        ax.margins(x=0.5)
        ax = _adjust_plt(ax=ax, **kwargs)

        if legend_position:
            ax.legend(loc=legend_position)

        if show:
            plt.show(block=False)
        else:
            return ax

    def er(
        self,
        ax=None,
        show: bool = True,
        cond_labels: tuple = ("Compatible", "Incompatible"),
        legend_labels: tuple = ("Observed", "Predicted"),
        legend_position: str = "upper left",
        **kwargs,
    ):
        """Plot error rate."""
        if ax is None:
            fig, ax = plt.subplots()

        kwargs.setdefault("ylabel", "Error Rate [%]")
        kwargs.setdefault("colors", ("black", "grey"))
        kwargs.setdefault("linestyles", ("-", "--"))
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)

        l_kws = _filter_dict(kwargs, plt.Line2D)
        for idx, res in enumerate((self.res_ob, self.res_th)):
            ax.plot(
                cond_labels,
                res.summary["per_err"],
                color=kwargs["colors"][idx],
                linestyle=kwargs["linestyles"][idx],
                label=legend_labels[idx],
                **l_kws,
            )

        kwargs.setdefault(
            "ylim",
            [
                0,
                np.max(self.res_ob.summary["per_err"]) + 5,
            ],
        )

        ax.margins(x=0.5)
        ax = _adjust_plt(ax=ax, **kwargs)

        if legend_position:
            ax.legend(loc=legend_position)

        if show:
            plt.show(block=False)
        else:
            return ax

    def rt_error(
        self,
        ax=None,
        show: bool = True,
        cond_labels: tuple = ("Compatible", "Incompatible"),
        legend_labels: tuple = ("Observed", "Predicted"),
        legend_position: str = "upper left",
        **kwargs,
    ):
        """Plot error RT's."""
        if ax is None:
            fig, ax = plt.subplots()

        kwargs.setdefault("ylabel", "RT Errors [ms]")
        kwargs.setdefault("colors", ("black", "grey"))
        kwargs.setdefault("linestyles", ("-", "--"))
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)

        l_kws = _filter_dict(kwargs, plt.Line2D)
        for idx, res in enumerate((self.res_ob, self.res_th)):
            ax.plot(
                cond_labels,
                res.summary["rt_err"],
                color=kwargs["colors"][idx],
                linestyle=kwargs["linestyles"][idx],
                label=legend_labels[idx],
                **l_kws,
            )

        min_rt, max_rt = (
            np.min(self.res_ob.summary["rt_err"]) - 100,
            np.max(self.res_ob.summary["rt_err"]) + 100,
        )
        kwargs.setdefault(
            "ylim",
            [
                550 if np.isnan(min_rt) else min_rt,
                550 if np.isnan(max_rt) else max_rt,
            ],
        )

        ax.margins(x=0.5)
        ax = _adjust_plt(ax=ax, **kwargs)

        if legend_position:
            ax.legend(loc=legend_position)

        if show:
            plt.show(block=False)
        else:
            return ax

    def cdf(
        self,
        ax=None,
        show: bool = True,
        legend_labels: tuple = (
            "Compatible Observed",
            "Incompatible Observed",
            "Compatible Predicted",
            "Incompatible Predicted",
        ),
        legend_position: str = "lower right",
        colors: tuple = ("green", "red"),
        **kwargs,
    ):
        """Plot CDF."""
        if ax is None:
            fig, ax = plt.subplots()

        kwargs.setdefault("linestyle", "None")
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)
        kwargs.setdefault("ylabel", "CDF")

        l_kws = _filter_dict(kwargs, plt.Line2D)
        for idx, comp in enumerate(("mean_comp", "mean_incomp")):
            ax.plot(
                self.res_ob.delta[comp],
                np.linspace(0, 1, self.res_ob.n_delta + 2)[1:-1],
                color=colors[idx % 2],  # Use modulo to cycle through colors
                label=legend_labels[idx],
                **l_kws,
            )

        kwargs["linestyle"] = "-"
        kwargs["marker"] = "None"

        l_kws = _filter_dict(kwargs, plt.Line2D)
        for idx, comp in enumerate(("mean_comp", "mean_incomp")):
            ax.plot(
                self.res_th.delta[comp],
                np.linspace(0, 1, self.res_th.n_delta + 2)[1:-1],
                color=colors[idx % 2],  # Use modulo to cycle through colors
                label=legend_labels[idx + 2],
                **l_kws,
            )

        kwargs.setdefault(
            "xlim",
            [
                np.min(self.res_ob.delta.mean_bin) - 100,
                np.max(self.res_ob.delta.mean_bin) + 100,
            ],
        )

        ax.margins(x=0.5)
        ax = _adjust_plt(ax=ax, **kwargs)

        if legend_position:
            ax.legend(loc=legend_position)

        if show:
            plt.show(block=False)
        else:
            return ax

    def caf(
        self,
        ax=None,
        show: bool = True,
        legend_labels: tuple = (
            "Compatible Observed",
            "Incompatible Observed",
            "Compatible Predicted",
            "Incompatible Predicted",
        ),
        legend_position: str = "lower right",
        colors: tuple = ("green", "red"),
        **kwargs,
    ):
        """Plot CAF."""
        if ax is None:
            fig, ax = plt.subplots()

        kwargs.setdefault("linestyle", "None")
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)
        kwargs.setdefault("xlabel", "RT Bin")
        kwargs.setdefault("ylabel", "CAF")

        l_kws = _filter_dict(kwargs, plt.Line2D)
        for idx, comp in enumerate(("comp", "incomp")):
            ax.plot(
                self.res_ob.caf["bin"],
                self.res_ob.caf[comp],
                color=colors[idx % 2],  # Use modulo to cycle through colors
                label=legend_labels[idx],
                **l_kws,
            )

        kwargs["linestyle"] = "-"
        kwargs["marker"] = "None"
        l_kws = _filter_dict(kwargs, plt.Line2D)
        for idx, comp in enumerate(("comp", "incomp")):
            ax.plot(
                self.res_th.caf["bin"],
                self.res_th.caf[comp],
                color=colors[idx % 2],  # Use modulo to cycle through colors
                label=legend_labels[idx + 2],
                **l_kws,
            )

        kwargs.setdefault("ylim", (0, 1.01))

        ax.set_xticks(range(1, self.res_th.n_caf + 1))
        ax.set_xticklabels([str(x) for x in range(1, self.res_th.n_caf + 1)])
        ax = _adjust_plt(ax=ax, **kwargs)

        if legend_position:
            ax.legend(loc=legend_position)

        if show:
            plt.show(block=False)
        else:
            return ax

    def delta(
        self,
        ax=None,
        show: bool = True,
        legend_labels: tuple = ("Observed", "Predicted"),
        legend_position: str = "lower right",
        **kwargs,
    ):
        """Plot reaction-time delta plots."""
        if ax is None:
            fig, ax = plt.subplots()

        kwargs.setdefault("color", "black")
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)
        kwargs.setdefault("linestyle", "None")
        kwargs.setdefault("xlabel", "Time (ms)")
        kwargs.setdefault("ylabel", r"$\Delta$ RT [ms]")

        l_kws = _filter_dict(kwargs, plt.Line2D)
        ax.plot(
            self.res_ob.delta["mean_bin"],
            self.res_ob.delta["mean_effect"],
            label=legend_labels[0],
            **l_kws,
        )

        kwargs["linestyle"] = "-"
        kwargs["marker"] = "None"
        l_kws = _filter_dict(kwargs, plt.Line2D)
        ax.plot(
            self.res_th.delta["mean_bin"],
            self.res_th.delta["mean_effect"],
            label=legend_labels[1],
            **l_kws,
        )

        kwargs.setdefault(
            "xlim",
            [
                np.min(self.res_ob.delta["mean_bin"]) - 100,
                np.max(self.res_ob.delta["mean_bin"]) + 100,
            ],
        )
        kwargs.setdefault(
            "ylim",
            [
                np.min(
                    [
                        np.min(self.res_ob.delta["mean_effect"]),
                        np.min(self.res_th.delta["mean_effect"])
                    ]
                ) - 25,
                np.max(
                    [
                        np.max(self.res_ob.delta["mean_effect"]),
                        np.max(self.res_th.delta["mean_effect"])
                    ]
                ) + 25,
            ],
        )

        ax = _adjust_plt(ax=ax, **kwargs)

        if legend_position:
            ax.legend(loc=legend_position)

        if show:
            plt.show(block=False)
        else:
            return ax

    def delta_errors(
        self,
        ax=None,
        show: bool = True,
        legend_labels: tuple = ("Observed", "Predicted"),
        legend_position: str = "upper right",
        **kwargs,
    ):
        """Plot error-rate delta plots."""
        if ax is None:
            fig, ax = plt.subplots()

        kwargs.setdefault("color", "black")
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)
        kwargs.setdefault("linestyle", "None")
        kwargs.setdefault("xlabel", "RT Bin")
        kwargs.setdefault("ylabel", r"$\Delta$ ER [%]")

        l_kws = _filter_dict(kwargs, plt.Line2D)
        ax.plot(
            self.res_ob.caf["bin"],
            self.res_ob.caf["effect"] * 100,  # Convert to percentage
            label=legend_labels[0],
            **l_kws,
        )

        kwargs["linestyle"] = "-"
        kwargs["marker"] = "None"
        l_kws = _filter_dict(kwargs, plt.Line2D)
        ax.plot(
            self.res_th.caf["bin"],
            self.res_th.caf["effect"] * 100,  # Convert to percentage
            label=legend_labels[1],
            **l_kws,
        )

        kwargs.setdefault(
            "ylim",
            [
                np.min(self.res_ob.caf["effect"]) * 100 - 5,  # Convert to percentage
                np.max(self.res_ob.caf["effect"]) * 100 + 5,  # Convert to percentage
            ],
        )

        ax.set_xticks(range(1, self.res_th.n_caf + 1))
        ax.set_xticklabels([str(x) for x in range(1, self.res_th.n_caf + 1)])
        ax = _adjust_plt(ax=ax, **kwargs)

        if legend_position:
            ax.legend(loc=legend_position)

        if show:
            plt.show(block=False)
        else:
            return ax
