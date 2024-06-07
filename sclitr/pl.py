from __future__ import annotations
from tqdm import tqdm
from scanpy import AnnData

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pylab as pl
import matplotlib.pyplot as plt

def clone(
    adata: AnnData,
    clone_col: str,
    clone_name: str,
    frameon: bool = False,
    s: float = 30,
    kwargs_background: dict = {},
    kwargs_clone: dict = {},
    clone_color: str = "black",
    ax: matplotlib.axes.Axes | None = None,
    title: str = None,
    return_fig: bool = False,
    basis: str = "X_umap",
) -> None | matplotlib.figure.Figure:
    """
    Plots single clone on gene expression embedding.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    clone_col : str
        Name of the column in `adata.obs` with clonal information.
    clone_name : str
        Name of the clone of interest.
    frameon : bool, optional
        `frameon` parameter of `sc.pl.embedding`, by default False.
    s : float, optional
        Dot size of the target clone on scatterplot, by default 30.
    kwargs_background : dict, optional
        Parameters of background plotting, by default {}.
    kwargs_clone : dict, optional
        Parameters of clone plotting, by default {}.
    clone_color : str, optional
        Color of the clone, by default "black".
    ax : matplotlib.axes.Axes | None, optional
        Matploitlib Axes object to plot on, by default None.
    title : str, optional
        Title, by default None.
    return_fig : bool, optional
        If true, returns Matplotlib Figure object, by default False.
    basis : str, optional
        Coordinates from `adata.obsm` to plot on, by default "X_umap".

    Returns
    -------
    None | matplotlib.figure.Figure
        If `return_fig == True` returns Matplotlib Figure object, else None.
    """
    kwargs_background = kwargs_background.copy()
    kwargs_clone = kwargs_clone.copy()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        return_fig = False

    sc.pl.embedding(
        adata,
        basis=basis,
        ax=ax,
        show=False,
        frameon=frameon,
        **kwargs_background
    )

    if "s" not in kwargs_clone:
        kwargs_clone["s"] = s

    if title is None:
        title = clone_name

    if "title" not in kwargs_clone:
        if "color" not in kwargs_clone:
            kwargs_clone["title"] = title
        else:
            kwargs_clone["title"] = title + "\n(" + kwargs_clone["color"] + ")"

    if return_fig:
        return fig

    sc.pl.embedding(
        adata[adata.obs[clone_col] == clone_name],
        basis=basis,
        ax=ax,
        show=False,
        frameon=frameon,
        na_color=clone_color,
        **kwargs_clone,
    )
    

def kde(
    adata: AnnData,
    groupby: str,
    group: str,
    basis: str = "X_umap",
    bw_method: float | str = 0.1,
    ax: matplotlib.axes.Axes | None = None,
    title: str | None = None,
    return_fig: bool = False,
    cmap: "ColorMap" = pl.cm.Reds,
    **kwargs,
) -> None | matplotlib.figure.Figure:
    """
    Plots kernel density estimation (KDE) of the group of cells calculated
    via `scipy.stats.gaussian_kde`.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    groupby : str
        Name of the column from `adata.obs` containing group of interest.
    group : str
        Group of interest from `adata.obs[groupby]`.
    basis : str, optional
        Slot from `adata.obsm` to plot, by default "X_umap".
    bw_method : float | str, optional
        The method used to calculate the estimator bandwidth (see more in `scipy.stats.gaussian_kde`
        documentation), by default 0.1.
    ax : matplotlib.axes.Axes | None, optional
        Matploitlib Axes object to plot on, by default None.
    title : str | None, optional
        Title, by default None.
    return_fig : bool, optional
        If true, returns Matplotlib Figure object, by default False.
    cmap : ColorMap, optional
        Color scheme of KDE plot, by default pl.cm.Reds.

    Returns
    -------
    None | matplotlib.figure.Figure
        If `return_fig == True` returns Matplotlib Figure object, else None.
    """
    from scipy.stats import gaussian_kde
    from matplotlib.colors import ListedColormap

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        return_fig = False

    if sum(adata.obs[groupby] == group) < 5:
        sc.pl.embedding(
            adata,
            basis=basis,
            ax=ax,
            show=False,
            frameon=False,
            s=5,
            legend_loc=None,
            title=title,
        )
    else:
        kernel = gaussian_kde(
            adata[adata.obs[groupby] == group].obsm[basis].T,
            bw_method=bw_method,
        )

        xmin = min(adata.obsm[basis].T[0])
        xmax = max(adata.obsm[basis].T[0])
        ymin = min(adata.obsm[basis].T[1])
        ymax = max(adata.obsm[basis].T[1])

        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = np.reshape(kernel(positions).T, xx.shape)

        my_cmap = cmap(np.arange(cmap.N))
        alphas = np.linspace(0, 1, cmap.N)

        BG = np.asarray([1., 1., 1.,])

        for i in range(cmap.N):
            my_cmap[i,:-1] = my_cmap[i,:-1] * alphas[i] + BG * (1. - alphas[i])

        my_cmap = ListedColormap(my_cmap)

        adata.obs["_"] = "black"
        adata.uns["__colors"] = ["#111111"]
        if title is None:
            title = f"{group} KDE"
        sc.pl.embedding(
            adata,
            basis=basis,
            ax=ax,
            show=False,
            frameon=False,
            s=5,
            color="_",
            legend_loc=None,
            title=title,
        )
        cset = ax.contour(xx, yy, f, colors="black", linewidths=1)
        cset = ax.contourf(xx, yy, f, cmap=my_cmap, alpha=0.8)

        del adata.obs["_"]
        del adata.uns["__colors"]
    if return_fig:
        return fig

def epochs_loss(
    clones: AnnData,
    uns_key: str = "clone2vec_mean_loss",
    return_fig: bool = False,
) -> None | matplotlib.figure.Figure:
    """
    Plots mean loss per epoch to see clone2vec learning process

    Parameters
    ----------
    clones : AnnData
        Annotated data matrix (clone level).
    uns_key : str, optional
        Key in `clones.uns` where mean loss per epoch is stored, by default "clone2vec_mean_loss".
    return_fig : bool, optional
        If true, returns Matplotlib Figure object, by default False, by default False.

    Returns
    -------
    None | matplotlib.figure.Figure
        If `return_fig == True` returns Matplotlib Figure object, else None.
    """
    import seaborn as sns
    
    fig, axes = plt.subplots(ncols=2, figsize=(8, 3))

    loss = np.array(clones.uns["clone2vec_mean_loss"])

    sns.lineplot(
        x=range(len(loss)),
        y=loss,
        ax=axes[0],
    )

    axes[0].set_xlabel("Number of epoch")
    axes[0].set_ylabel("Mean loss")
    axes[0].grid(alpha=0.3)

    sns.lineplot(
        x=range(len(loss) - 1),
        y=loss[:-1] - loss[1:],
        ax=axes[1],
    )

    axes[1].set_xlabel("Number of epoch")
    axes[1].set_ylabel("Δ(Mean loss)")
    axes[1].grid(alpha=0.3)
    axes[1].set_yscale("log")

    fig.tight_layout()

    if return_fig:
        return fig
    

def basic_stats(
    adata: AnnData,
    obs_name: str,
    return_fig: bool = False,
    title: str = "Clone size distribution",
) -> None | matplotlib.figure.Figure:
    """
    Plots basic QC plots for clonal experiment.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    obs_name : str
        Name of the column in `adata.obs` with clonal information.
    return_fig : bool, optional
        If true, returns Matplotlib Figure object, by default False, by default False, by default False.
    title : str, optional
        Title, by default "Clone size distribution".

    Returns
    -------
    None | matplotlib.figure.Figure
        If `return_fig == True` returns Matplotlib Figure object, else None.
    """
    import seaborn as sns
    fig, axes = plt.subplots(ncols=2, figsize=(8, 4))

    sns.histplot(
        adata.obs[obs_name].value_counts()[1:],
        bins=30,
        log=True,
        alpha=1,
        edgecolor="black",
        ax=axes[0],
    )
    axes[0].grid(alpha=0.3)
    axes[0].set_xlabel("Clone size")
    axes[0].set_ylabel("Number of clones")

    size_dist = adata.obs[obs_name].value_counts()[1:].value_counts().sort_index()

    sns.lineplot(np.cumsum(size_dist[::-1]), ax=axes[1])
    axes[1].grid(alpha=0.3)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Clone size")
    axes[1].set_ylabel("Number of clones bigger than this")

    plt.suptitle(title)
    fig.tight_layout()
    
    if return_fig:
        return fig
    
def double_injection_composition(
    adata: AnnData,
    early_injection: str,
    late_injection: str,
    min_clone_size: int = 5,
    non_clonal_str: str = "NA",
) -> None:
    """
    Plots composition of clones with with multuply injected cells. In ideal clonal reconstruction
    we're going to see Russian doll-like picture: smaller clones will be inside bigger ones.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    early_injection : str
        Column with clonal labelling from early injection (bigger clones).
    late_injection : str
        Column with clonal labelling from late injection (smaller clones).
    min_clone_size : int, optional
        Minimal size of the clone to include in the plot, by default 5.
    non_clonal_str : str, optional
        Values of unlabelled cells, by default "NA".
    """
    import mpltern

    early_injection = "E7.5:clones"
    late_injection = "E8.5:clones"
    non_clonal_str = "NA"

    possible_combinations = adata.obs[[
        early_injection, late_injection,
    ]][(adata.obs[[early_injection, late_injection]] != non_clonal_str).sum(axis=1) == 2]
    possible_combinations = list(set([(i, j) for i, j in possible_combinations.values]))

    early_clone_sizes = []
    late_clone_sizes = []
    di_clone_sizes = []

    for early_clone, late_clone in possible_combinations:
        early = sum(adata.obs[early_injection] == early_clone)
        late = sum(adata.obs[late_injection] == late_clone)
        both = sum((adata.obs[late_injection] == late_clone) & (adata.obs[early_injection] == early_clone))
        if (early >= min_clone_size) and (late >= min_clone_size):
            early_clone_sizes.append(early - both)
            late_clone_sizes.append(late - both)
            di_clone_sizes.append(both)

    ax = plt.subplot(projection="ternary", ternary_sum=100)

    ax.set_tlabel("Both barcodes (%)")
    ax.set_llabel("Early barcode (%)")
    ax.set_rlabel("Late barcode (%)")

    # Order = top, left, right
    ax.scatter(
        di_clone_sizes,
        early_clone_sizes,
        late_clone_sizes,
        s=10,
        color="grey",
        edgecolor="k",
        linewidth=0.5,
    )

    ax.grid(alpha=0.3)

    ax.set_tlim(-3, 103)
    ax.set_llim(-3, 103)
    ax.set_rlim(-3, 103)