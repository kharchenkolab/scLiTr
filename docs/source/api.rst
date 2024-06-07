.. module:: sclitr

API
===

How to import **scLiTr** package::

   import sclitr as sl

Datasets
-------------

.. autosummary::
   :toctree: .

   datasets.Weinreb_in_vitro
   datasets.Erickson_murine_development

Preprocessing
-------------

.. autosummary::
   :toctree: .

   pp.prepare_clones2cells
   pp.prepare_multiple_injections

Tools
-----

.. autosummary::
   :toctree: .

   tl.clonal_nn
   tl.clone2vec
   tl.transfer_clonal_annotation
   tl.summarize_expression
   tl.refill_ct

Plotting
-----

.. autosummary::
   :toctree: .

   pl.basic_stats
   pl.double_injection_composition
   pl.clone
   pl.epochs_loss
   pl.kde
   