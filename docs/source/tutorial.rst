Tutorial
========

Step 1: Clonal nearest neighbours graph construction
****************************************************

Firstly, we have to identify *k* nearest clonally labelled cells for each cell. It will create "bag of clones"
(similar to "bag of words") that will be used for *clone2vec* training further.

.. code-block:: python

    import scanpy as sc
    import sclitr as sl

    sl.tl.clonal_nn(
        adata,
        obs_name="clone", # Column with clonal labels
        use_rep="X_pca", # Which dimred use for graph construction
        min_size=5, # Minimal clone size
    )

Minimal clone size parameter is used to exclude small clones from embedding construction.

Step 2: clone2vec
*****************

Now, we have to train our neural network to predict clonal labels of nearest neighbours for each
clonally labelled cell.

.. code-block:: python

    clones = sl.tl.clone2vec(
        adata,
        obs_name="clone", # Column with clonal labels
        fill_ct="cell_type", # Column with cell type annotation to fill `clones.X`
    )

After execution of this function we have AnnData-object :code:`clones` with clonal vector representation
stored in :code:`clones.obsm["clone2vec"]`. Now we can work with it like with regular scRNA-Seq dataset.

Step 3: clone2vec analysis
**************************

.. code-block:: python

    sc.pp.neighbors(clones, use_rep="word2vec")
    sc.tl.umap(clones)
    sc.tl.leiden(clones)

And after perform all other additional steps of analysis.

For more detailed walkthrough see **Exampels** section.