from __future__ import annotations
from tqdm import tqdm
from scanpy import AnnData

import scanpy as sc
import numpy as np
import pandas as pd


def clonal_nn(
    adata: AnnData,
    obs_name: str,
    k: int = 15,
    non_clonal_str: str = "NA",
    use_rep: str = "X_pca",
    tqdm_bar: bool = False,
    min_size: int = 10,
    random_state: None | int = 4,
    copy: bool = False,
    obsm_name: str = "bag-of-clones",
    **kwargs,
) -> None | AnnData:
    """
    Function to find top k clonally labelled cells for each cell.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    obs_name : str
        Name of the column in `adata.obs` with clonal information.
    k : int, optional
        Number of nearest neighbours in the graph, by default 15.
    non_clonal_str : str, optional
        Which value is used to indicate absence of the clonal information for the cell, by default "NA".
    use_rep : str, optional
        Based on which representation kNN graph should be built, by default "X_pca".
    tqdm_bar : bool, optional
        Set to `True` if you want to see the progress bar, by default False.
    min_size : int, optional
        Clones with size less than `min_size` will be considered as cells without clonal labelling,
        by default 10.
    random_state : None | int, optional
        Random state, by default 4.
    copy : bool, optional
        Determines whether a copy is returned, by default False.
    obsm_name : str, optional
        Name of newly created graph slot in `adata.obsm`, by default "bag-of-clones".

    Returns
    -------
    AnnData
        Returns or updates `adata`, depending on `copy`.
    """
    from scipy.sparse import csr_matrix
    import pynndescent
    
    # Removing clones with small size
    clonal_obs = adata.obs[obs_name].copy()
    clones_counts = clonal_obs.value_counts()
    small_clones = clones_counts[clones_counts < min_size].index
    clonal_obs = pd.Series([
        clone if clone not in small_clones else non_clonal_str for clone in clonal_obs
    ]).astype(str).astype("category")

    var_mapping = dict(zip(
        clonal_obs.cat.categories[clonal_obs.cat.categories != non_clonal_str],
        range(len(clonal_obs.cat.categories[clonal_obs.cat.categories != non_clonal_str])),
    ))
    
    train = adata[clonal_obs != non_clonal_str].obsm[use_rep]
    obs_col = clonal_obs[clonal_obs != non_clonal_str].astype(str)
    obs_col.index = range(len(obs_col))
    test = adata.obsm[use_rep]
    index = pynndescent.NNDescent(train, random_state=random_state, **kwargs)
    index.prepare()
    neighbors = index.query(test, k=k)[0]
    
    col_ind = []
    row_ind = []
    data = []

    for i in (tqdm(range(len(neighbors))) if tqdm_bar else range(len(neighbors))):
        nn = obs_col[neighbors[i]].value_counts()
        nn = nn[nn > 0]
        col_ind += [var_mapping[var] for var in nn.index]
        row_ind += [i] * len(nn)
        data += list(nn.values)

    if copy:
        adata = adata.copy()

    adata.obsm[obsm_name] = csr_matrix((data, (row_ind, col_ind)))
    adata.uns[obsm_name + "_names"] = list(var_mapping.keys())

    adata_clonal = sc.AnnData(
        X=csr_matrix((data, (row_ind, col_ind))),
        obs=pd.DataFrame(index=adata.obs_names),
        var=pd.DataFrame(index=list(var_mapping.keys())),
    )
    
    if copy:
        return adata

def clone2vec(
    adata: AnnData,
    obs_name: str,
    z_dim: int = 10,
    n_epochs: int = 100,
    batch_size: int = 64,
    device: str = "cpu",
    fill_ct: None | str = None,
    tqdm_bar: bool = True,
    obsm_name: str = "bag-of-clones",
    obsm_key: str = "word2vec",
    random_state: None | int = 4,
) -> AnnData:
    """
    Clonal embedding construction, for more info see [PMID: None].

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    obs_name : str
        Name of the column in `adata.obs` with clonal information.
    z_dim : int, optional
        Number of hidden dimensions, by default 10.
    n_epochs : int, optional
        Number of epochs in training, by default 100.
    batch_size : int, optional
        Batch size in training, by default 64.
    device : str, optional
        Where neural network training should be performed, by default "cpu".
    fill_ct : None | str, optional
        Name of the column in `adata.obs` with cell type labels to fill
        `clones.X`, by default None.
    tqdm_bar : bool, optional
        Set to `True` if you want to see the progress bar, by default True.
    obsm_name : str, optional
        Slot in `adata.obsm` with clonal kNN graph, by default "bag-of-clones".
    obsm_key : str, optional
        Slot in `clones.obsm` with vector representation of the clones,
        by default "word2vec".
    random_state : None | int, optional
        Random state, by default 4.

    Returns
    -------
    AnnData
        Annotated data matrix with contains clonal representation.
    """
    from .word2vec import SkipGram
    from tqdm import tqdm

    import random
    import torch
    import torch.nn as nn

    torch.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)

    clone2idx = dict(zip(
        adata.uns[f"{obsm_name}_names"],
        range(len(adata.uns[f"{obsm_name}_names"])),
    ))
    idx2clone = dict(zip(clone2idx.values(), clone2idx.keys()))
    indices = np.array(range(len(adata.uns[f"{obsm_name}_names"])))

    k_estimated = list(set(adata.obsm[obsm_name].A.sum(axis=1)))
    if len(k_estimated) > 1:
        raise Exception("adata.obsm[obsm_name] should contain the result of kNN graph construction with fixed k.")
    k_estimated = int(k_estimated[0])

    adata_only_clones = adata[adata.obs[obs_name].isin(adata.uns[f"{obsm_name}_names"])]
    pairs = []
    for X, clone in zip(
        adata_only_clones.obsm[obsm_name].A,
        adata_only_clones.obs[obs_name],
    ):
        cl_nn = X > 0
        pairs += list(zip([clone2idx[clone]] * k_estimated, indices[cl_nn]))
    pairs = np.array(pairs)

    train_loader = torch.utils.data.DataLoader(
        pairs,
        batch_size=batch_size,
        shuffle=True,
    )

    model = SkipGram(
        z_dim=z_dim,
        vocab_size=len(indices),
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.NLLLoss()

    epochs_mean_loss = []
    for epoch in (tqdm(range(n_epochs)) if tqdm_bar else range(n_epochs)):
        losses = []
        for batch_idx, data in enumerate(train_loader):
            model.train()
            
            x_batch = data[:, 0].to(device)
            y_batch = data[:, 1].to(device)
            
            log_ps = model(x_batch)
            loss = criterion(log_ps, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        epochs_mean_loss.append(np.mean(losses))

    clone2vec = model.embedding.weight.data.cpu().numpy()
    if not (fill_ct is None):
        cell_counts = adata_only_clones.obs.groupby(
            [fill_ct, obs_name]
        ).size().unstack()[adata_only_clones.uns[f"{obsm_name}_names"]]
        
        var_names = list(cell_counts.index)
        obs_names = list(cell_counts.columns)
        
        cell_counts = cell_counts.values
        freqs = cell_counts / cell_counts.sum(axis=0)
    else:
        var_names = ["None"]
        obs_names = adata.uns[f"{obsm_name}_names"]

        cell_counts = np.matrix([0] * len(adata.uns[f"{obsm_name}_names"]))
        freqs = np.matrix([0] * len(adata.uns[f"{obsm_name}_names"]))

    clones = sc.AnnData(
        X=cell_counts.T,
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=var_names),
        layers={
            "frequencies": freqs.T,
            "counts": cell_counts.T,
        },
        obsm={
            obsm_key: clone2vec.copy(),
        },
        uns={
            "clone2vec_mean_loss": epochs_mean_loss,
        },
    )
    clones.obs["n_cells"] = clones.X.sum(axis=1)

    return clones

def transfer_clonal_annotation(
    adata: AnnData,
    clones: AnnData,
    adata_clone_name: str,
    adata_obs_name: str,
    clones_obs_name: str,
    fill_values: str = "NA",
) -> None:
    """
    Transfer labels from clonal to cells AnnData objects.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix at cell level.
    clones : AnnData
        Annotated data matrix at clone level.
    adata_clone_name : str
        Name of the column in `adata.obs` with clonal information.
    adata_obs_name : str
        Name of the newly generated column in `adata.obs` with transferred
        labels from clonal object.
    clones_obs_name : str
        Name of the column in `clones.obs` with labels to transfer.
    fill_values : str, optional
        Value to fill for unlabelled cells, by default "NA".
    """
    
    clone_mapping = dict(clones.obs[clones_obs_name])
    adata.obs[adata_obs_name] = [
        clone_mapping[clone] if clone in clone_mapping else fill_values
        for clone in adata.obs[adata_clone_name]
    ]
    adata.obs[adata_obs_name] = adata.obs[adata_obs_name].astype("category")

def summarize_expression(
    adata: AnnData,
    clones: AnnData,
    obs_name: str = "clone",
    strategy: str = "average",
    layer: str | None = None,
    use_raw: bool | None = None,
    subset_obs: str | None = None,
    target_value: str | None = None,
) -> AnnData:
    """
    Function adds gene expression information for each clone from
    an object with clonal embeddings.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix at cell level with gene expressions stored.
    clones : AnnData
        Annotated data matrix at clone level.
    obs_name : str
        Name of the column in `adata.obs` with clonal information, by defaul "clone".
    strategy : str
        Strategy that is used for summarizing gene expression information per clone:
        either "sum" or "average", by default "average".
    layer : str | None, optional
        If not `None`, `adata.layers[layer]` will be used to summarize expression,
        by default None.
    use_raw : bool | None, optional
        If `adata.raw.X` should be used for gene expression summary, by default None.
    subset_obs : str | None, optional
        Column in `adata.obs` to perform subsetting while summing expression
        (for example, column with cell type label), by default None.
    target_value : str | None, optional
        Value in `adata.obs[subset_obs]` to perform subsetting while summing expression
        (for example, some specific cell type), by default None.

    Returns
    -------
    AnnData
        Annotated data matrix at clone level with gene expression stored in `clones.X`.
    """
    from scipy.sparse import csr_matrix
    
    if use_raw and not (layer is None):
        raise Exception(f"Can't use both `adata.raw` and `adata.layers['{layer}']`")
    if strategy not in ("average", "sum"):
        raise Exception(f"Only average and sum methods of expression aggregation are supported")
    
    if subset_obs is None:
        mask = np.array([True] * len(adata))
    else:
        mask = np.array(adata.obs[subset_obs] == target_value)

    if not (layer is None):
        X = adata.layers[layer]
        var_names = adata.var_names
    elif use_raw or not (adata.raw is None):
        X = adata.raw.X
        var_names = adata.raw.var_names
    else:
        X = adata.X
        var_names = adata.var_names

    clones_expr = {}
    for clone in clones.obs_names:
        if strategy == "average":
            try:
                clones_expr[clone] = X[(adata.obs[obs_name] == clone) & mask].mean(axis=0).A[0]
            except ZeroDivisionError:
                clones_expr[clone] = np.zeros(len(var_names))
        else:
            clones_expr[clone] = X[(adata.obs[obs_name] == clone) & mask].sum(axis=0).A[0]
    
    clones_expr = sc.AnnData(pd.DataFrame(clones_expr, index=var_names).T)
    
    clones_expr.X = csr_matrix(clones_expr.X)
    clones_expr.uns = clones.uns.copy()
    clones_expr.obs = clones.obs.copy()
    clones_expr.obsm = clones.obsm.copy()
    clones_expr.obsp = clones.obsp.copy()
    
    return clones_expr

def refill_ct(
    adata: AnnData,
    obs_name: str,
    ct_col: str,
    clones: AnnData,
) -> AnnData:
    """
    This function creates new annotated data matrix at clone level with new proportions in `clones.X`
    (e. g. if you decided to change level of cell type annotation).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (cell level).
    obs_name : str
        Name of the column in `adata.obs` with clonal information.
    ct_col : str
        Name of the column in `adata.obs` with new cell type labels.
    clones : AnnData
        Annotated data matrix (clone level).

    Returns
    -------
    AnnData
        New annotated data matrix at clone level.
    """
    clones_new = sc.AnnData(adata.obs.groupby(
        [ct_col, obs_name]
    ).size().unstack().T.loc[clones.obs_names])
    
    clones_new.uns = clones.uns.copy()
    clones_new.obs = clones.obs.copy()
    clones_new.obsm = clones.obsm.copy()
    clones_new.obsp = clones.obsp.copy()
    
    clones_new.layers["counts"] = clones_new.X.copy()
    clones_new.layers["frequencies"] = (clones_new.X.T / clones_new.X.sum(axis=1)).T
    
    return clones_new
