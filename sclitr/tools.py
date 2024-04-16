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
) -> AnnData:
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
        range(len(clonal_obs.cat.categories) - 1),
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
    obsm_key: str = "word2vec",
    random_state: None | int = 4,
) -> AnnData:
    from .word2vec import SkipGram
    from tqdm import tqdm

    import random
    import torch
    import torch.nn as nn

    torch.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)

    clone2idx = dict(zip(
        adata.uns["bag-of-clones_names"],
        range(len(adata.uns["bag-of-clones_names"])),
    ))
    idx2clone = dict(zip(clone2idx.values(), clone2idx.keys()))
    indices = np.array(range(len(adata.uns["bag-of-clones_names"])))

    k_estimated = list(set(adata.obsm["bag-of-clones"].A.sum(axis=1)))
    if len(k_estimated) > 1:
        raise Exception("adata.obsm['bag-of-clones'] should contain the result of kNN graph construction with fixed k.")
    k_estimated = int(k_estimated[0])

    adata_only_clones = adata[adata.obs[obs_name].isin(adata.uns["bag-of-clones_names"])]
    pairs = []
    for X, clone in zip(
        adata_only_clones.obsm["bag-of-clones"].A,
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
        cell_counts = adata_only_clones.obs.groupby([fill_ct, obs_name]).size().unstack()[adata_only_clones.uns["bag-of-clones_names"]]
        freqs = cell_counts / cell_counts.sum(axis=0)
    else:
        cell_counts = freqs = np.array([0] * len(adata.uns["bag-of-clones_names"]))

    clones = sc.AnnData(
        X=cell_counts.T,
        layers={
            "frequencies": freqs.values.T,
            "counts": cell_counts.values.T,
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
    
    clone_mapping = dict(clones.obs[clones_obs_name])
    adata.obs[adata_obs_name] = [
        clone_mapping[clone] if clone in clone_mapping else fill_values
        for clone in adata.obs[adata_clone_name]
    ]
    adata.obs[adata_obs_name] = adata.obs[adata_obs_name].astype("category")

def summarize_expression(
    adata: AnnData,
    obs_name: str,
    clones: AnnData,
    type: str = "average",
    layer: str | None = None,
    use_raw: bool | None = None,
    subset_obs: str | None = None,
    target_value: str | None = None,
) -> AnnData:
    from scipy.sparse import csr_matrix
    
    if use_raw and not (layer is None):
        raise Exception(f"Can't use both `adata.raw` and `adata.layers['{layer}']`")
    if type not in ("average", "sum"):
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
        if type == "average":
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
