from __future__ import annotations
from tqdm import tqdm
from scanpy import AnnData

import scanpy as sc
import numpy as np
import pandas as pd

def prepare_clones2cells(
    adata: AnnData,
    embedding_type: str,
    clonal_obs: None | str = None,
    keep_obs: None | list = None,
    dimred_name: str = "X_umap",
) -> pd.DataFrame:
    """
    Function that prepares dataframes for clones2cells viewer.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with gene expression or clonal data.
    embedding_type : str
        One of "clone2vec" or "GEX" — which dataset is provided to generate
        clones2cells-friendly dataframe.
    clonal_obs : None | str, optional
        Please provide name of the column with clonal labelling in the case
        of gene expression object, by default None.
    keep_obs : None | list, optional
        Which columns from `adata.obs` should be kept, by default None.
    dimred_name : str, optional
        Name of `adata.obsm` slot to keep, by default "X_umap".

    Returns
    -------
    pd.DataFrame
        clones2cells-friendly dataframe.
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
    Function that prepares clone2vec-friendly object in the case of multiple injections.
    Briefly, it duplicates cells with more than one clonal labelling.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    injection_cols : list[str]
        Columns in `adata.obs` with clonal labellings.
    na_value : str, optional
        Which value is used to indicate absence of the clonal information for the cell,
        by default "NA".
    final_obs_name : str, optional
        Name of the new column in `adata.obs` with newly generated clonal labelling annotation,
        by default "clone".

    Returns
    -------
    AnnData
        Newly generated Annotated data matrix with duplicated cells with multiple clonal
        assignments.
    """
    
    bc_list = []
    clone_obs = []

    for bc, clonal_labels in adata.obs[injection_cols].iterrows():
        labeled_cells = clonal_labels[clonal_labels != na_value]
        if len(labeled_cells) == 0:
            bc_list.append(bc)
            clone_obs.append(na_value)
        else:
            for label, clone in clonal_labels[clonal_labels != na_value].iteritems():
                bc_list.append(bc)
                clone_obs.append(label + "_" + clone)
                
    adata_demult = adata[bc_list]
    adata_demult.obs[final_obs_name] = clone_obs
    adata_demult.obs_names_make_unique()
    
    return adata_demult