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
    if return_fig:
        return fig

def epochs_loss(
    clones: AnnData,
    uns_key: str = "clone2vec_mean_loss",
    return_fig: bool = False,
) -> None | matplotlib.figure.Figure:
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
