from __future__ import annotations
from scanpy import AnnData
from catboost import CatBoostRegressor, CatBoostClassifier
from pandas import DataFrame

import scanpy as sc
import numpy as np
import pandas as pd
import torch.nn as nn


def calculate_shap_correlation(
    shapdata_ct: AnnData,
) -> None:
    """
    Calculate correlation between gene expression and SHAP values for each cell type.

    Parameters
    ----------
    shapdata_ct : AnnData
        AnnData object with `.X` containing gene expression and `.layers` containing SHAP matrices
        with keys of the form 'shap_<celltype>'.
    
    Returns
    -------
    None
        Adds new columns to `shapdata_ct.var` named 'shap_corr_<celltype>' containing correlation
        coefficients for each gene.
    """
    cts = [i[5:] for i in shapdata_ct.layers.keys() if i[:5] == "shap_"]
    expr_matrix = shapdata_ct.X.A

    for ct in cts:
        if f"shap_corr_{ct}" not in shapdata_ct.var_names:
            shap_matrix = shapdata_ct.layers[f"shap_{ct}"].A
            correlations = []
            
            for i in range(expr_matrix.shape[1]):
                correlations.append(np.corrcoef(expr_matrix[:, i], shap_matrix[:, i])[0, 1])
                
            shapdata_ct.var[f"shap_corr_{ct}"] = correlations
            shapdata_ct.var[f"shap_corr_{ct}"] = shapdata_ct.var[f"shap_corr_{ct}"].fillna(0)


class SkipGram(nn.Module):
    """
    A SkipGram model implemented in PyTorch for word embedding.

    Parameters
    ----------
    z_dim : int
        Dimensionality of the embedding space.
    vocab_size : int
        Size of the vocabulary.
    device : torch.device
        Device to run the model on (e.g., 'cpu' or 'cuda').

    Attributes
    ----------
    embedding : nn.Embedding
        Embedding layer for input words.
    output : nn.Linear
        Linear layer mapping embeddings to vocabulary space.
    log_softmax : nn.LogSoftmax
        LogSoftmax activation over output layer.

    Methods
    -------
    forward(input_word)
        Perform forward pass and return log-probabilities for context words.
    """
    import torch.nn as nn
    def __init__(self, z_dim, vocab_size, device):
        super(SkipGram, self).__init__()
        self.device = device

        self.embedding = nn.Embedding(
            vocab_size,
            z_dim,
        )
        self.output = nn.Linear(
            z_dim,
            vocab_size,
        )
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_word):
        emb_input = self.embedding(input_word)
        context_scores = self.output(emb_input)
        log_ps = self.log_softmax(context_scores)
        
        return log_ps
    

def split_test_eval(
    clones: AnnData,
    use_rep: str = "clone2vec",
    fraction: float = 0.2,
    add_label: str | None = "eval_set",
    random_state: None | int = 4,
) -> tuple[list, list]:
    """
    Split data into training and validation using geometric sketching.

    Parameters
    ----------
    clones : AnnData
        AnnData object containing clone embeddings in `.obsm[use_rep]`.
    use_rep : str, optional
        Key in `.obsm` to use for representation, by default "clone2vec".
    fraction : float, optional
        Fraction of cells to use for validation, by default 0.2.
    add_label : str or None, optional
        If provided, adds a categorical column in `.obs` marking train/validation, by default "eval_set".
    random_state : int or None, optional
        Seed for reproducibility, by default 4.

    Returns
    -------
    tuple[list, list]
        Lists of training and validation cell names.
    """
    from geosketch import gs

    idx = gs(
        X=clones.obsm[use_rep],
        N=np.round(len(clones) * fraction).astype(int),
        seed=random_state,
    )
    validation = clones[idx].obs_names.copy()
    train = clones.obs_names[
        ~clones.obs_names.isin(validation)
    ].copy()

    if add_label:
        clones.obs[add_label] = [
            "validation" if i in validation else "train"
            for i in clones.obs_names
        ]
        clones.obs[add_label] = clones.obs[add_label].astype("category")
        
    return list(train), list(validation)


def run_CatBoostRegressor(
    train_expr: DataFrame,
    train_c2v: DataFrame,
    validation_expr: DataFrame,
    validation_c2v: DataFrame,
    use_gpu: bool = False,
    num_trees: int = 10000,
    early_stopping_rounds: int = 100,
    verbose: bool = True,
    random_state: None | int = 4,
) -> CatBoostRegressor:
    """
    Train a CatBoostRegressor on expression data to predict continuous clone2vec features.

    Parameters
    ----------
    train_expr : DataFrame
        Training input features.
    train_c2v : DataFrame
        Training targets (clone2vec vectors).
    validation_expr : DataFrame
        Validation input features.
    validation_c2v : DataFrame
        Validation targets (clone2vec vectors).
    use_gpu : bool, optional
        Whether to use GPU, by default False.
    num_trees : int, optional
        Maximum number of boosting iterations, by default 10000.
    early_stopping_rounds : int, optional
        Early stopping rounds, by default 100.
    verbose : bool, optional
        Verbosity of CatBoost training, by default True.
    random_state : int or None, optional
        Random seed, by default 4.

    Returns
    -------
    CatBoostRegressor
        Trained CatBoostRegressor model.
    """
    from catboost import Pool, CatBoostRegressor
    
    if "celltype_column" in train_expr.columns:
        train = Pool(
            data=train_expr,
            label=train_c2v,
            cat_features=["celltype_column"],
        )
        validation = Pool(
            data=validation_expr,
            label=validation_c2v,
            cat_features=["celltype_column"],
        )
    else:
        train = Pool(data=train_expr, label=train_c2v)
        validation = Pool(data=validation_expr, label=validation_c2v)
        
    if use_gpu:
        task_type = "GPU"
    else:
        task_type = "CPU"
        
    model = CatBoostRegressor(
        loss_function="MultiRMSE",
        eval_metric="MultiRMSE",
        num_trees=num_trees,
        early_stopping_rounds=early_stopping_rounds,
        task_type=task_type,
        boosting_type="Plain",
        verbose=verbose,
        random_seed=random_state,
    )
    
    model.fit(train, eval_set=validation, use_best_model=True)
    
    return model


def run_CatBoostClassifier(
    train_expr: DataFrame,
    train_ct: DataFrame,
    validation_expr: DataFrame,
    validation_ct: DataFrame,
    use_gpu: bool = False,
    num_trees: int = 10000,
    early_stopping_rounds: int = 100,
    verbose: bool = True,
    random_state: None | int = 4,
) -> CatBoostClassifier:
    """
    Train a CatBoostClassifier to predict discrete cell types.

    Parameters
    ----------
    train_expr : DataFrame
        Training input features.
    train_ct : DataFrame
        Training labels (cell types).
    validation_expr : DataFrame
        Validation input features.
    validation_ct : DataFrame
        Validation labels (cell types).
    use_gpu : bool, optional
        Whether to use GPU, by default False.
    num_trees : int, optional
        Maximum number of boosting iterations, by default 10000.
    early_stopping_rounds : int, optional
        Early stopping rounds, by default 100.
    verbose : bool, optional
        Verbosity of CatBoost training, by default True.
    random_state : int or None, optional
        Random seed, by default 4.

    Returns
    -------
    CatBoostClassifier
        Trained CatBoostClassifier model.
    """
    from catboost import Pool, CatBoostClassifier
    
    if "celltype_column" in train_expr.columns:
        train = Pool(
            data=train_expr,
            label=train_ct,
            cat_features=["celltype_column"],
        )
        validation = Pool(
            data=validation_expr,
            label=validation_ct,
            cat_features=["celltype_column"],
        )
    else:
        train = Pool(data=train_expr, label=train_ct)
        validation = Pool(data=validation_expr, label=validation_ct)
        
    if use_gpu:
        task_type = "GPU"
    else:
        task_type = "CPU"
        
    model = CatBoostClassifier(
        loss_function="MultiCrossEntropy",
        eval_metric="MultiCrossEntropy",
        num_trees=num_trees,
        early_stopping_rounds=early_stopping_rounds,
        task_type=task_type,
        verbose=verbose,
        random_seed=random_state,
    )
    
    model.fit(train, eval_set=validation, use_best_model=True)
    
    return model


def prepare_expr(
    adata: AnnData,
    clone_col: str = "clone",
    use_raw: bool = False,
    layer: str | None = None,
    ct_col: str | None = None,
) -> DataFrame:
    """
    Extract expression data from AnnData and format it for model training.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing single-cell data.
    clone_col : str, optional
        Column in `.obs` that identifies clones, by default "clone".
    use_raw : bool, optional
        Whether to use `.raw` expression values, by default False.
    layer : str or None, optional
        Layer to use if specified, by default None.
    ct_col : str or None, optional
        Column in `.obs` specifying cell types, added as metadata.

    Returns
    -------
    DataFrame
        DataFrame containing gene expression with optional clone and celltype annotations.
    """
    if use_raw and layer:
        raise Exception("Only one of `use_raw` or `layer` should be used!")
    
    if use_raw:
        expr = adata.raw.to_adata().to_df()
    elif layer:
        expr = adata.layers[layer].A if "sparse" in str(type(adata.layers[layer])) else adata.layers[layer]
        expr = pd.DataFrame(expr, index=adata.obs_names, columns=adata.var_names)
    else:
        expr = adata.to_df()
    
    expr["clone_column"] = adata.obs[clone_col].astype(str)
    if ct_col:
        expr["celltype_column"] = adata.obs[ct_col]
    
    return expr


def prepare_pseudobulk(
    expr: DataFrame,
    pseudobulk: bool = True,
    ct_col: str | None = None,
    limit_ct: str | None = None,
    use_ct: bool = False
) -> DataFrame:
    """
    Aggregate single-cell data into pseudobulk expression profiles.

    Parameters
    ----------
    expr : DataFrame
        Expression DataFrame with 'clone_column' and optionally 'celltype_column'.
    pseudobulk : bool, optional
        Whether to aggregate into pseudobulk, by default True.
    ct_col : str or None, optional
        Cell type column name, used for grouping.
    limit_ct : str or list or None, optional
        Restrict to specific cell type(s), by default None.
    use_ct : bool, optional
        Whether to retain celltype_column in output, by default False.

    Returns
    -------
    DataFrame
        Aggregated expression data at the (cell type, clone) level if pseudobulk is True,
        otherwise unmodified input.
    """
    if pseudobulk:
        if ct_col is None:
            expr = expr.groupby("clone_column").mean()
        else:
            expr = expr.groupby(["celltype_column", "clone_column"]).mean().reset_index().dropna()
            expr.index = expr.celltype_column.astype(str) + ":" + expr.clone_column.astype(str)
    
    if isinstance(limit_ct, str):
        expr = expr[expr.celltype_column == limit_ct]
    elif isinstance(limit_ct, list):
        expr = expr[expr.celltype_column.isin(limit_ct)]
    
    if not use_ct and ct_col:
        del expr["celltype_column"]
    
    return expr