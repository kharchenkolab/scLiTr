scLiTr — Python package for single cell lineage tracing analysis
================================================================

**scLiTr** is a Python package for analysis of lineage tracing coupled with single-cell RNA-Seq.

The main key of the package are *clonal embeddings* — vector representations of the whole clones in low dimensional space (*clone2vec*). These
representations is a dropout-robust and cluster-free way of representation of heterogeneity within clonal behaviour for cell type tree-free
hypothesis generation regarding cells' multipotency.

*clone2vec* builds representation of clones in exact same way with popular word embedding algorithm — *word2vec* — via construction two-layers
fully connected neural network (specifically it uses Skip-Gram architecture) that aims to predict neighbour cells clonal labellings by clonal label
of cells. As a result, clones that exist in similar context in gene expression space will have similar weights in this neural network, and these
weights will be used as embedding for further analysis.

Installation
------------

scLiTr package might be installed via pip:

.. code-block:: console

   pip install sclitr

or the latest development version can be installed from GitHub using:

.. code-block:: console

   pip install git+https://github.com/kharchenkolab/scLiTr

.. toctree::
   :caption: Main
   :maxdepth: 2
   :hidden:

   tutorial
   api

.. toctree::
   :caption: Exampels
   :maxdepth: 0
   :hidden:

   Basic_usage
   Multiple_injections