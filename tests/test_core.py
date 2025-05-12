import numpy as np
import scanpy as sc
import pytest

import sys
sys.path.append("/home/sergey/tools/scLiTr/")
import sclitr as sl

@pytest.fixture(scope="session")
def adata():
    return sc.read_h5ad("tests/Weinreb_subsampled.h5ad")

@pytest.fixture(scope="session")
def adata_filtered(adata):
    sl.pp.filter_clones(adata, clonal_obs="clone", min_size=30, max_size=150)
    adata = adata[adata.obs.clone != "NA"]
    return adata

@pytest.fixture(scope="session")
def adata_clonal_nn(adata_filtered):
    sl.tl.clonal_nn(adata_filtered, obs_name="clone")
    return adata_filtered
    
@pytest.fixture(scope="session")
def clones(adata_clonal_nn):
    return sl.tl.clone2vec(adata_clonal_nn, obs_name="clone", n_epochs=10, fill_ct="cell_type", z_dim=2)

@pytest.fixture(scope="session")
def adata_multiple(adata_filtered):
    adata_multiple = adata_filtered.copy()
    adata_multiple.obs["clone_1"] = adata_multiple.obs.clone.copy()
    adata_multiple.obs["clone_2"] = adata_multiple.obs.clone.copy()
    adata_multiple = sl.pp.prepare_multiple_injections(adata_multiple, injection_cols=["clone_1", "clone_2"],
                                                       final_obs_name="clone_combined")
    return adata_multiple

@pytest.fixture(scope="session")
def shapdata_c2v_process(adata_filtered, clones):
    shapdata = sl.tl.predict_c2v(adata_filtered, clones, clone_col="clone", ct_col="cell_type",
                                 limit_ct="Undiff", num_trees=100, verbose=False, use_ct=False)
    return (shapdata, clones)

@pytest.fixture(scope="session")
def shapdata_ct_process(adata_filtered, shapdata_c2v_process):
    shapdata_c2v, clones = shapdata_c2v_process
    shapdata = sl.tl.predict_ct(adata_filtered, clones, clone_col="clone", ct_col="cell_type", ct_layer="frequencies",
                                limit_ct="Undiff", num_trees=100, verbose=False, use_ct=False)
    return (shapdata_c2v, shapdata, clones)

@pytest.fixture(scope="session")
def clones_expr(adata_filtered, clones):
    return sl.tl.summarize_expression(adata_filtered, clones)

@pytest.fixture(scope="session")
def transfer_clonal_annotation(adata_filtered, shapdata_ct_process):
    shapdata_c2v, shapdata_ct, clones = shapdata_ct_process
    sl.tl.transfer_clonal_annotation(adata_filtered, clones, "clone", "eval_set", "eval_set")
    return (adata_filtered, clones)

@pytest.fixture(scope="session")
def clones_refill(adata_filtered, clones):
    return sl.tl.refill_ct(adata_filtered, "clone", "clone", clones)

def test_filter_clones(adata_filtered):
    assert len(set(adata_filtered.obs.clone)) == 58
    
def test_multiple_injections(adata_multiple):
    assert len(set(adata_multiple.obs.clone_combined)) == 116
    
def test_clonal_nn(adata_clonal_nn):
    assert "bag-of-clones" in adata_clonal_nn.obsm
    assert "bag-of-clones_names" in adata_clonal_nn.uns
    assert (adata_clonal_nn.obsm["bag-of-clones"].sum(axis=1) == 15).all()
    
def test_clones(clones):
    assert "n_cells" in clones.obs
    assert clones.obs.n_cells.sum() == 3960
    assert "clone2vec_mean_loss" in clones.uns
    assert "clone2vec" in clones.obsm
    assert clones.X.sum() == 3960
    assert "frequencies" in clones.layers
    assert (np.round(clones.layers["frequencies"].sum(axis=1), 1) == 1).all()
    assert "counts" in clones.layers
    assert (clones.layers["counts"] == clones.X).all()
    
def test_shapdata_c2v(shapdata_c2v_process):
    shapdata_c2v, clones = shapdata_c2v_process
    assert "shap" in shapdata_c2v.layers
    assert "shap_c2v0" in shapdata_c2v.layers
    assert "shap_c2v1" in shapdata_c2v.layers
    assert len(shapdata_c2v) == 57
    assert "clone2vec_predicted" in clones.uns
    assert "clone2vec_predicted_names" in clones.uns
    assert len(clones.uns["clone2vec_predicted_names"]) == len(shapdata_c2v)
    assert len(clones.uns["clone2vec_predicted_names"]) == clones.uns["clone2vec_predicted"].shape[0]
    assert "eval_set" in clones.obs
    
def test_shapdata_ct(shapdata_ct_process):
    shapdata_c2v, shapdata_ct, clones = shapdata_ct_process
    assert "shap" in shapdata_ct.layers
    for ct in clones.var_names:
        assert f"shap_{ct}" in shapdata_ct.layers
    assert len(shapdata_ct) == 57
    assert "ct_predicted" in clones.uns
    assert "ct_predicted_names" in clones.uns
    assert len(clones.uns["ct_predicted_names"]) == len(shapdata_ct)
    assert len(clones.uns["ct_predicted_names"]) == clones.uns["ct_predicted"].shape[0]
    assert "eval_set" in clones.obs
    
def test_clones_expr(clones_expr, adata_filtered):
    assert len(clones_expr.var_names) == len(adata_filtered.var_names)
    assert not (clones_expr.X < 0).toarray().any()
    
def test_transfer_clonal_annotation(transfer_clonal_annotation):
    adata, clones = transfer_clonal_annotation
    assert sum(adata.obs.eval_set == "validation") == clones[clones.obs.eval_set == "validation"].obs.n_cells.sum()
    
def test_refill_ct(clones_refill):
    assert ((clones_refill.X > 0).sum(axis=1) == 1).all()
    
def test_clones2cells_gex(adata_filtered):
    assert len(set(sl.pp.prepare_clones2cells(adata_filtered, "GEX", clonal_obs="clone").clone)) == 58
    
def test_clones2cells_c2v(clones):
    assert len(sl.pp.prepare_clones2cells(clones, "clone2vec", dimred_name="clone2vec")) == 58
    
def test_basic_stats(adata):
    sl.pl.basic_stats(adata, "clone")
    
def test_double_injection_plot(adata_multiple):
    sl.pl.double_injection_composition(adata_multiple, early_injection="clone_1", late_injection="clone_2")
    
def test_clone_plot(adata):
    sl.pl.clone(adata, "clone", "clone_1070")
    
def test_loss_plot(clones):
    sl.pl.epochs_loss(clones)
    
def test_kde_plot(adata):
    sl.pl.kde(adata, "clone", "clone_1070")
    
def test_ct_markers_plot(shapdata_ct_process):
    shapdata_c2v, shapdata_ct, clones = shapdata_ct_process
    sl.pl.ct_predictors(shapdata_ct, "Undiff")
    
def test_c2v_annotation_plot(shapdata_ct_process):
    shapdata_c2v, shapdata_ct, clones = shapdata_ct_process
    sl.pl.c2v_annotation(shapdata_c2v, shapdata_ct)
