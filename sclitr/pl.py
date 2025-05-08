from __future__ import annotations
from typing import Tuple
from tkinter import N
from tqdm import tqdm
from scanpy import AnnData
from .utils import calculate_shap_correlation

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pylab as pl
import matplotlib.pyplot as plt


def c2v_annotation(
    shapdata_c2v: AnnData,
    shapdata_ct: AnnData,
    top_n: int = 20,
    figsize: Tuple[float, float] = (8.0, 8.0),
    cmap_barplot: str = "Reds",
    cmap_heatmap: str = "RdBu_r",
    normalize: bool = True,
) -> None:
    """
    Visualizes SHAP-based feature importance from clone2vec and cell-type-specific models.

    This function generates a two-panel figure:
    - Left: Bar plot of the top `top_n` features (genes) by average SHAP value from the clone2vec model.
    - Right: Heatmap showing sign-corrected, cell-type-specific SHAP contributions for the same features,
      scaled per cell type if `normalize=True`.

    Parameters
    ----------
    shapdata_c2v : AnnData
        Annotated data object containing clone2vec SHAP values in `shapdata_c2v.layers["shap"]`.
    shapdata_ct : AnnData
        Annotated data object containing cell-type-specific SHAP values in
        `shapdata_ct.layers["shap_<celltype>"]` and corresponding correlations in
        `shapdata_ct.var["shap_corr_<celltype>"]`.
    top_n : int, optional
        Number of top genes (by mean absolute SHAP value in clone2vec model) to display, by default 20.
    figsize : Tuple[float, float], optional
        Size of the output figure, by default (8.0, 8.0).
    cmap_barplot : str, optional
        Colormap used for the barplot, by default "Reds".
    cmap_heatmap : str, optional
        Colormap used for the heatmap, by default "RdBu_r".
    normalize : bool, optional
        Whether to normalize heatmap values column-wise (per cell type), by default True.

    Returns
    -------
    None
        The function displays the plot and does not return any value.
    """
    import warnings
    import seaborn as sns

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if "average_shap" not in shapdata_c2v.var.columns:
            shapdata_c2v.var["average_shap"] = shapdata_c2v.layers["shap"].sum(axis=0).A[0]
        top_genes = shapdata_c2v.var.average_shap.sort_values(ascending=False)[:top_n]
        cts = [i.split("_")[1] for i in list(shapdata_ct.layers.keys()) if "shap_" in i]

        fig, axes = plt.subplots(ncols=2, figsize=figsize, gridspec_kw={"width_ratios": [1, 3]})

        # Plotting of the clone2vec SHAP values

        # Create a colorscheme
        norm = plt.Normalize(0, top_genes.max())
        bar_colors = plt.cm.get_cmap(cmap_barplot)(norm(top_genes))[::-1]

        sns.barplot(
            x=top_genes,
            y=top_genes.index,
            ax=axes[0],
            hue=top_genes.values,
            edgecolor="k",
            palette=bar_colors,
        )
        axes[0].grid(alpha=0.3)
        axes[0].set_xlabel("Mean |SHAP|")
        axes[0].set_ylabel("")
        axes[0].legend_.remove()
        axes[0].set_title("clone2vec model")
        
        for spine in axes[0].spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)
            spine.set_edgecolor("k")

        heatmap = {}

        # Check if all correlations were calculated
        calculate_shap_correlation(shapdata_ct)

        for ct in cts:
            heatmap[ct] = (
                np.abs(shapdata_ct[:, top_genes.index].layers[f"shap_{ct}"]).mean(axis=0).A[0] *
                shapdata_ct[:, top_genes.index].var[f"shap_corr_{ct}"].values
            )
            
        heatmap = pd.DataFrame(heatmap, index=top_genes.index)
        if normalize:
            heatmap = heatmap / np.abs(heatmap).max(axis=0)
        
        sns.heatmap(heatmap, cmap=cmap_heatmap, vmin=-heatmap.max().max(), ax=axes[1],
                    linecolor="k", linewidth=0.5, cbar_kws={"shrink": 0.5})
        axes[1].set_yticks([], [])
        
        for spine in axes[1].spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)
            spine.set_edgecolor("k")
            
        cbar = axes[1].collections[0].colorbar

        # Style the colorbar's border (spines)
        for spine in cbar.ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)
            spine.set_edgecolor("k")
            
        axes[1].set_title("Cell type model")
        if normalize:
            cbar.set_label("Normalized sign-corrected SHAP")
        else:
            cbar.set_label("Sign-corrected SHAP")
        
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01)


def ct_predictors(
    shapdata_ct: AnnData,
    celltype: str,
    top_n: int = 10,
    title: str | None = None,
    ax: None = None,
    return_fig: bool = False,
) -> None | matplotlib.figure.Figure:
    """
    Visualizes the importance and directionality of gene predictors for a given cell type's abundance,
    based on SHAP values.

    The plot shows the mean absolute SHAP values (importance) against their correlation with gene
    expression, helping to identify positive and negative drivers of cell type proportions.

    Parameters
    ----------
    shapdata_ct : AnnData
        AnnData object containing SHAP values, typically from `sl.tl.predict_ct`.
    celltype : str
        Target cell type to analyze.
    top_n : int, optional
        Number of top predictor genes to highlight, by default 10.
    title : str | None, optional
        Plot title. If None, uses the cell type name.
    ax : matplotlib.axes.Axes | None, optional
        Axes object to draw the plot on. If None, a new figure is created.
    return_fig : bool, optional
        If True, returns the Matplotlib Figure object. Default is False.

    Returns
    -------
    None | matplotlib.figure.Figure
        Returns Figure if `return_fig` is True; otherwise returns None.
    """
    import textalloc as ta
    import seaborn as sns
    
    if f"shap_{celltype}" not in shapdata_ct.layers:
        raise Exception("Shapley values for this cell type weren't calculated.")
        
    # Step 1: calculate correlation
    if f"shap_corr_{celltype}" not in shapdata_ct.var.columns:
        shap_matrix = shapdata_ct.layers[f"shap_{celltype}"].A
        expr_matrix = shapdata_ct.X.A
        correlations = []
        
        for i in range(expr_matrix.shape[1]):
            correlations.append(np.corrcoef(expr_matrix[:, i], shap_matrix[:, i])[0, 1])
            
        shapdata_ct.var[f"shap_corr_{celltype}"] = correlations
        shapdata_ct.var[f"shap_corr_{celltype}"] = shapdata_ct.var[f"shap_corr_{celltype}"].fillna(0)
    
    # Plot correlations
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        return_fig = False
        
    df = pd.DataFrame({
        "shap_correlation": shapdata_ct.var[f"shap_corr_{celltype}"].values,
        "mean_shap_mag": np.abs(shapdata_ct.layers[f"shap_{celltype}"]).mean(axis=0).A[0],
    }, index=shapdata_ct.var_names)

    sns.scatterplot(
        x="shap_correlation",
        y="mean_shap_mag",
        data=df,
        edgecolor=None,
        s=5,
        color="grey",
        ax=ax,
    )

    # Top-predictors
    df_subset = df.loc[df.mean_shap_mag.sort_values()[-top_n:].index]
    
    sns.scatterplot(
        x="shap_correlation",
        y="mean_shap_mag",
        data=df_subset[df_subset.shap_correlation > 0],
        edgecolor="black",
        s=20,
        color=sns.color_palette()[2],
        ax=ax,
    )
    
    sns.scatterplot(
        x="shap_correlation",
        y="mean_shap_mag",
        data=df_subset[df_subset.shap_correlation < 0],
        edgecolor="black",
        s=20,
        color=sns.color_palette()[3],
        ax=ax,
    )
    
    ta.allocate(
        ax,
        x=df_subset.shap_correlation,
        y=df_subset.mean_shap_mag,
        text_list=df_subset.index,
        x_scatter=df.shap_correlation,
        y_scatter=df.mean_shap_mag,
        textsize=10,
        linecolor="grey",
    )

    ax.set_xlabel("corr(SHAP, expr)")
    ax.set_ylabel("mean(|SHAP|)")
    ax.grid(alpha=0.3)
    if title is None:
        ax.set_title(celltype)
    else:
        ax.set_title(title)
    
    if return_fig:
        return fig


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
    Plots the spatial distribution of a single clone on a 2D embedding.

    First plots all cells as a background, then overlays the selected clone using a distinct color.

    Parameters
    ----------
    adata : AnnData
        AnnData object with cell metadata and embeddings.
    clone_col : str
        Column in `adata.obs` indicating clonal identity.
    clone_name : str
        Clone to be highlighted.
    frameon : bool, optional
        Whether to show axes frame in the plot. Default is False.
    s : float, optional
        Dot size for the highlighted clone. Default is 30.
    kwargs_background : dict, optional
        Additional plotting arguments for the background cells.
    kwargs_clone : dict, optional
        Additional plotting arguments for the highlighted clone.
    clone_color : str, optional
        Color to use for the highlighted clone. Default is "black".
    ax : matplotlib.axes.Axes | None, optional
        Axes object to draw the plot on. If None, a new figure is created.
    title : str, optional
        Plot title. If None, uses the clone name.
    return_fig : bool, optional
        If True, returns the Matplotlib Figure object. Default is False.
    basis : str, optional
        Embedding key in `adata.obsm` to use for coordinates. Default is "X_umap".

    Returns
    -------
    None | matplotlib.figure.Figure
        Returns Figure if `return_fig` is True; otherwise returns None.
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
    Plots a kernel density estimate (KDE) of cells from a specified group on a 2D embedding.

    Uses `scipy.stats.gaussian_kde` to estimate density of a group and overlays it as a contour plot.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing embeddings and group annotations.
    groupby : str
        Column in `adata.obs` used to identify cell groups.
    group : str
        Specific group to visualize.
    basis : str, optional
        Embedding key from `adata.obsm` to use. Default is "X_umap".
    bw_method : float | str, optional
        Bandwidth for the KDE. Passed to `scipy.stats.gaussian_kde`. Default is 0.1.
    ax : matplotlib.axes.Axes | None, optional
        Axes object to draw the plot on. If None, a new figure is created.
    title : str | None, optional
        Plot title. If None, uses the group name with "KDE" suffix.
    return_fig : bool, optional
        If True, returns the Matplotlib Figure object. Default is False.
    cmap : ColorMap, optional
        Colormap to use for KDE overlay. Default is `pl.cm.Reds`.

    Returns
    -------
    None | matplotlib.figure.Figure
        Returns Figure if `return_fig` is True; otherwise returns None.
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
    Plot the mean loss per epoch and its change across epochs during clone2vec training.

    Parameters
    ----------
    clones : AnnData
        Annotated data matrix containing training statistics in `clones.uns`.
    uns_key : str, optional
        Key in `clones.uns` that stores a list of mean losses per epoch.
        Default is "clone2vec_mean_loss".
    return_fig : bool, optional
        If True, returns the Matplotlib Figure object. Default is False.

    Returns
    -------
    None or matplotlib.figure.Figure
        The plot figure if `return_fig` is True, otherwise None.

    Raises
    ------
    KeyError
        If `uns_key` is not present in `clones.uns`.
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
    Plot basic statistics of clone size distribution.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing clone annotations in `.obs`.
    obs_name : str
        Column name in `adata.obs` that stores clone identifiers.
    return_fig : bool, optional
        If True, returns the Matplotlib Figure object. Default is False.
    title : str, optional
        Title of the plot. Default is "Clone size distribution".

    Returns
    -------
    None or matplotlib.figure.Figure
        The plot figure if `return_fig` is True, otherwise None.

    Notes
    -----
    Single-cell clone annotations with only one occurrence (clone size = 1) are excluded
    from both plots.
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
    Plot ternary composition of clones labeled by two different injections.

    In ideal clonal reconstructions, smaller clones from a later injection
    should nest within larger clones from an earlier injection — resembling
    a "Russian doll" pattern. This function visualizes the composition of
    shared and unique cells across early and late injections using a ternary plot.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with clonal labels in `.obs`.
    early_injection : str
        Column name in `.obs` containing clone IDs from the early injection.
    late_injection : str
        Column name in `.obs` containing clone IDs from the late injection.
    min_clone_size : int, optional
        Minimum number of cells required for a clone to be included in the plot.
        Default is 5.
    non_clonal_str : str, optional
        Value representing unlabelled or non-clonal cells. Default is "NA".

    Returns
    -------
    None

    Notes
    -----
    Each point in the ternary plot represents a combination of an early and late
    clone that overlap in at least `min_clone_size` cells. Axes correspond to
    the fraction of cells unique to early injection, unique to late injection,
    and shared between both (the intersection).
    """
    import mpltern

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