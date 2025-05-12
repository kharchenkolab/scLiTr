# Configuration file for the Sphinx documentation builder.

# -- Project information

project = "scLiTr"
copyright = "2024, Kharchenko lab, Adameyko lab"
authors = "Isaev, Kharchenko"

release = "1.0.1"
version = "1.0.1"

# -- General configuration ------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath("../../."))

import sclitr

needs_sphinx = "2.0"

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "numpydoc",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

numpydoc_xref_param_type = True
numpydoc_xref_ignore = {"optional", "type_without_description", "BadException"}

# Run docstring validation as part of build process
numpydoc_validation_checks = {"all", "GL01", "SA04", "RT03"}

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"
html_logo = "logo.png"
html_theme_options = {
    "titles_only": True,
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": False,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
    "display_version": False,
    "logo_only": True,
}

# Generate the API documentation when building
autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
napoleon_custom_sections = [("Params", "Parameters")]
