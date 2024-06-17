<p align="center"><a href="https://sclitr.readthedocs.io/en/latest/"><img src="docs/source/logo.png" height="250"></a></p>

# scLiTr

scLiTr is a Python package for analysis of lineage tracing coupled with single-cell RNA-Seq.

The main key of the package are clonal embeddings — vector representations of the whole clones
in low dimensional space (clone2vec). These representations is a dropout-robust and cluster-free
way of representation of heterogeneity within clonal behaviour for cell type tree-free hypothesis
generation regarding cells' multipotency.

clone2vec builds representation of clones in exact same way with popular word embedding algorithm — word2vec —
via construction two-layers fully connected neural network (specifically it uses Skip-Gram architecture) that
aims to predict neighbour cells clonal labellings by clonal label of cells. As a result, clones that exist in
similar context in gene expression space will have similar weights in this neural network, and these weights
will be used as embedding for further analysis.

## Installation

scLiTr might be installed via `pip`:
```bash
pip install sclitr
```

## Documentation

Please visit [documentation web-site](https://sclitr.readthedocs.io/en/latest/) to check out API description and a few
tutorials with analysis.

## clones2cells

For interactive exploration of clonal and gene expression embeddings together we recommend using
our simple tool [clones2cells](https://github.com/serjisa/clones2cells_app).
