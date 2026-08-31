# Sphinx configuration file.
import os

# 0. Add this directory to path (needed for custom autodoc2_docstrings_parser)
import sys
from importlib import metadata

current_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_script_dir)

# 1. Project information
project = "specfuel"
copyright = "2026 Troy N. Smith"
author = "Troy N. Smith"
release = metadata.version("specfuel")

# 2. General configuration
extensions = [
    "myst_parser",
    "autodoc2",
    "sphinx.ext.napoleon",
]
templates_path = []
exclude_patterns = []

# 3. Options for HTML output
html_theme = "pydata_sphinx_theme"
html_static_path = []
html_theme_options = {
    "github_url": "https://github.com/github-org-name/specfuel",
}

# 4. MyST configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]

# 5. Autodoc2 configuration
autodoc2_packages = [
    "../../src/specfuel",
    "../../src/fuellib",
]
autodoc2_render_plugin = "myst"
autodoc2_docstring_parser_regexes = [
    (r".*", "autodoc2_docstrings_parser"),
]
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = False
