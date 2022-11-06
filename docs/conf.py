# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys

sys.path.insert(0, ".")

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "graphtask"
copyright = "2022, David Muhr"
author = "David Muhr"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "numpydoc",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.autosectionlabel"
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autosectionlabel_prefix_document = True

# -- Options for autosummary -------------------------------------------------
# See: https://stackoverflow.com/questions/65198998/sphinx-warning-autosummary-stub-file-not-found-for-the-methods-of-the-class-c
autosummary_generate = False
numpydoc_class_members_toctree = False

# -- Options for sphinx gallery ----------------------------------------------
sphinx_gallery_conf = {
    "doc_module": "graphtask",
    "reference_url": {
        # The module you locally document uses None
        "graphtask": None,
    },
    "examples_dirs": "../examples",  # path to your example scripts
    "gallery_dirs": "_gallery",  # path to where to save gallery generated output
    "ignore_pattern": "skip_",
    "filename_pattern": "",  # process all files in ``examples``
    "image_scrapers": ("pygraphviz", "matplotlib"),
}

# -- Options for sphinx-copybutton -------------------------------------------
# https://sphinx-copybutton.readthedocs.io/en/latest/use.html
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

# -- Options for the Theme ---------------------------------------------------
# See: https://github.com/pydata/pydata-sphinx-theme/blob/main/docs/conf.py
html_theme_options = {
    "external_links": [
        {
            "url": "https://github.com/davnn/graphtask/releases",
            "name": "Changelog",
        }
    ],
    "github_url": "https://github.com/davnn/graphtask",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/graphtask",
            "icon": "fa-solid fa-box",
        }
    ],
    "logo": {
        "image_light": "logo-light.svg",
        "image_dark": "logo-dark.svg",
        "alt_text": "graphtask",
    },
    "announcement": "This is a young, community-supported project - check it out on <a href=https://github.com/davnn/graphtask>GitHub</a>. Your contributions are extremely welcome!",
}

# -- Doctest options ---------------------------------------------------------
# See: https://stackoverflow.com/questions/58189699/use-sphinx-doctest-with-example
doctest_global_setup = """
from graphtask import *
from graphtask.visualize import *
"""
