from __future__ import annotations
from tqdm import tqdm
from scanpy import AnnData

import scanpy as sc
import numpy as np
import pandas as pd

def filter_clones(
    adata: AnnData,
    na_value: str = "NA",
    clonal_obs: str = "Clone",
    min_size: int | None = None,
    max_size: int | None = None,
    inplace: bool = True,
) -> None | AnnData:
    """
    Filters clonal observations in the provided AnnData object based on specified size criteria.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing the clonal information in `adata.obs`.
    na_value : str, optional
        The value used to indicate absence of clonal labeling for a cell. Default is "NA".
    clonal_obs : str, optional
        The name of the column in `adata.obs` containing clonal labels. Default is "Clone".
    min_size : int, optional
        The minimum number of cells required to keep a clone. Default is None (no lower bound).
    max_size : int, optional
        The maximum number of cells allowed for a clone. Default is None (no upper bound).
    inplace : bool, optional
        Whether to modify the `adata` object in place or return a modified copy. Default is True.

    Returns
    -------
    None or AnnData
        If `inplace` is True, the function modifies the `adata` object directly and returns None.
        Otherwise, it returns a new `AnnData` object with the filtered clonal observations.
    """
    whitelist = adata.obs[clonal_obs].value_counts()
    whitelist = whitelist[whitelist.index != na_value]
    if not(min_size is None):
        whitelist = whitelist[whitelist >= min_size]
    if not(max_size is None):
        whitelist = whitelist[whitelist <= max_size]
    whitelist = whitelist.index
    
    if inplace:
        adata.obs[clonal_obs] = [
            i if i in whitelist else na_value for i in adata.obs[clonal_obs]
        ]
    else:
        adata = adata.copy()
        adata.obs[clonal_obs] = [
            i if i in whitelist else na_value for i in adata.obs[clonal_obs]
        ]
        return adata

def prepare_clones2cells(
    adata: AnnData,
    embedding_type: str,
    clonal_obs: None | str = None,
    keep_obs: None | list = None,
    dimred_name: str = "X_umap",
) -> pd.DataFrame:
    """
    Prepares a dataframe suitable for the `clones2cells` visualization, based on either clone2vec or gene expression (GEX) data.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing either clone2vec embeddings or gene expression data.
    embedding_type : str
        The type of embedding data to use. Should be either "clone2vec" or "GEX".
    clonal_obs : None | str, optional
        The name of the column in `adata.obs` containing clonal labels. Required for "GEX" embedding type.
    keep_obs : None | list, optional
        List of additional columns in `adata.obs` to retain in the output dataframe. Default is None.
    dimred_name : str, optional
        The name of the key in `adata.obsm` that contains the dimensionality-reduced embeddings. Default is "X_umap".

    Returns
    -------
    pd.DataFrame
        A dataframe containing the relevant columns for visualization in `clones2cells`, including UMAP coordinates and clonal labels.

    Raises
    ------
    Exception
        If `embedding_type` is not one of "clone2vec" or "GEX", or if `embedding_type` is "GEX" but `clonal_obs` is not provided.
    """
    keep_obs = keep_obs.copy()

    if embedding_type == "clone2vec":
        if keep_obs is None:
            df = pd.DataFrame(
                adata.obsm[dimred_name],
                index=adata.obs_names,
                columns=["UMAP1", "UMAP2"],
            )
        else:
            df = adata.obs[keep_obs]
            df = pd.concat([df, pd.DataFrame(
                adata.obsm[dimred_name],
                index=adata.obs_names,
                columns=["UMAP1", "UMAP2"],
            )], axis=1)
        return df
    
    elif embedding_type == "GEX":
        if clonal_obs is None:
            raise Exception("Please provide `clonal_obs` argument")
        elif keep_obs is None:
            keep_obs = [clonal_obs]
        else:
            keep_obs.append(clonal_obs)
        df = adata.obs[keep_obs]
        df.columns = list(df.columns[:-1]) + ["clone"]
        df = pd.concat([df, pd.DataFrame(
            adata.obsm[dimred_name],
            index=adata.obs_names,
            columns=["UMAP1", "UMAP2"],
        )], axis=1)
        return df
    
    else:
        raise Exception("`embedding_type` argument should be one of `clone2vec` or `GEX`")

def prepare_multiple_injections(
    adata: AnnData,
    injection_cols: list[str],
    na_value: str = "NA",
    final_obs_name: str = "clone",
) -> AnnData:
    """
    Prepares a clone2vec-friendly AnnData object by handling multiple clonal injections per cell.
    Duplicates cells with multiple clonal labels into separate rows for each unique clone label.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing the clonal labels across multiple columns.
    injection_cols : list[str]
        List of column names in `adata.obs` that contain the clonal labels.
    na_value : str, optional
        The value used to indicate missing or absent clonal information. Default is "NA".
    final_obs_name : str, optional
        The name of the new column in `adata.obs` to store the merged clone labels. Default is "clone".

    Returns
    -------
    AnnData
        A new AnnData object with cells duplicated where necessary, with updated clonal labeling in the `final_obs_name` column.
    """
    
    bc_list = []
    clone_obs = []

    for bc, clonal_labels in adata.obs[injection_cols].iterrows():
        labeled_cells = clonal_labels[clonal_labels != na_value]
        if len(labeled_cells) == 0:
            bc_list.append(bc)
            clone_obs.append(na_value)
        else:
            for label, clone in clonal_labels[clonal_labels != na_value].items():
                bc_list.append(bc)
                clone_obs.append(label + "_" + clone)
                
    adata_demult = adata[bc_list]
    adata_demult.obs[final_obs_name] = clone_obs
    adata_demult.obs_names_make_unique()
    
    return adata_demult