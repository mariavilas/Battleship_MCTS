import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # so Sphinx finds tfg/

# -- Project information -----------------------------------------------------
project = 'Battleship'
copyright = '2025, UO282899'
author = 'UO282899'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',      # for Google‐style docstrings
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
