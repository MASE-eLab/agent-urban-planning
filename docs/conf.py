"""Sphinx configuration for agent-urban-planning."""
from __future__ import annotations

import os
import sys
from datetime import date

# -- Path setup ----------------------------------------------------------
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -------------------------------------------------
project = "agent-urban-planning"
copyright = f"{date.today().year}, The agent-urban-planning Authors"
author = "TBD"

import agent_urban_planning  # noqa: E402

version = agent_urban_planning.__version__
release = version

# -- General configuration -----------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.inheritance_diagram",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
    "sphinx_codeautolink",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Use Markdown (MyST) AND reStructuredText.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- HTML output ---------------------------------------------------------
html_theme = "furo"
html_title = f"agent-urban-planning {version}"
html_static_path: list[str] = []  # add when we have logos/CSS overrides
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/MASE-eLab/agent-urban-planning/",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/MASE-eLab/agent-urban-planning",
            "html": "",
            "class": "fa-brands fa-github",
        },
    ],
}

# -- Autodoc / autosummary ----------------------------------------------
autosummary_generate = True
autosummary_imported_members = False
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
    "inherited-members": False,
    "undoc-members": False,  # only document things with docstrings
    "exclude-members": "__init__, __repr__, __str__, __weakref__, __dict__",
}
autodoc_typehints = "description"  # render type hints in the parameter description
autodoc_typehints_format = "short"  # use 'list[int]' not 'typing.List[int]'
typehints_fully_qualified = False
add_module_names = False  # display 'UtilityEngine' not 'aup.decisions.utility.UtilityEngine'

# -- Napoleon (Google-style docstrings) -----------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True  # include __init__ docs in class docstring
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_param = True
napoleon_use_ivar = True
napoleon_use_rtype = True

# -- Intersphinx (link to other libraries' docs) -------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- MyST (Markdown) ------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
    "attrs_inline",
    "smartquotes",
]
myst_heading_anchors = 3

# -- sphinx_copybutton ---------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- sphinx_codeautolink -------------------------------------------------
codeautolink_autodoc_inject = True
# Suppress "Could not match a code example to HTML" warnings raised when
# docstring Examples sections use commented-out lines (because the actual
# values like `params`, `agent`, `env` aren't bound in the doc-eval scope).
# These are cosmetic warnings — docstring rendering still works.
suppress_warnings = ["codeautolink.match_block", "codeautolink.match_name"]

# -- sphinx_design -------------------------------------------------------
# (no extra config needed)

# -- sphinxcontrib.mermaid -----------------------------------------------
mermaid_output_format = "raw"

# -- Misc ----------------------------------------------------------------
# Treat warnings as errors only on local builds; ReadTheDocs needs more
# leeway during initial bootstrapping.
nitpicky = False
