# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
# Add the project root so autodoc can find the 'models' package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# -- Project information -----------------------------------------------------

project = 'theory_idp_design'
copyright = '2026, Neha Tyagi, Jackson Boodry, Vita Chou, Wilton Snead, Krishna Shrinivas'
author = 'Neha Tyagi, Jackson Boodry, Vita Chou, Wilton Snead, Krishna Shrinivas'
release = '2026'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
]

# Napoleon settings (Google-style docstrings)
napoleon_google_docstrings = True
napoleon_numpy_docstrings = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

# Autodoc settings
autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}

# Mock imports that may not be available at doc-build time
autodoc_mock_imports = [
    'jax',
    'jaxlib',
    'optax',
    'numpy',
    'scipy',
    'matplotlib',
    'seaborn',
    'pandas',
    'tqdm',
]

templates_path = ['_templates']
exclude_patterns = []

language = 'en'

# -- Options for HTML output -------------------------------------------------

html_theme = 'alabaster'
html_static_path = ['_static']
html_theme_options = {
    'description': 'Physics-informed IDP sequence design via differentiable optimization',
    'github_user': '',
    'github_repo': 'theory_idp_design',
    'fixed_sidebar': True,
    'sidebar_width': '260px',
}

# -- Intersphinx mapping -----------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}
