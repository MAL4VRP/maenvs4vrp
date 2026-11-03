import glob
import os
import shutil
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Document the code in this repository, not an installed copy of the package
sys.path.insert(0, BASE_DIR)

# The tutorial notebooks live in maenvs4vrp/learning_notebooks (single source of truth);
# copy them into the docs source tree at build time.
NOTEBOOKS_SRC = os.path.join(BASE_DIR, 'maenvs4vrp', 'learning_notebooks')
NOTEBOOKS_DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'notebooks')
os.makedirs(NOTEBOOKS_DST, exist_ok=True)
for notebook in glob.glob(os.path.join(NOTEBOOKS_SRC, '*.ipynb')):
    target = os.path.join(NOTEBOOKS_DST, os.path.basename(notebook))
    if os.path.exists(target) and os.path.samefile(notebook, target):
        continue  # already linked to the source notebook
    shutil.copy2(notebook, target)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'maenvs4vrp'
copyright = '2026, maenvs4vrp'
author = 'maenvs4vrp'
release = '0.3.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'nbsphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_math_dollar',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx_copybutton']

nbsphinx_allow_errors = True
nbsphinx_execute = 'never'

mathjax3_config = {
  "tex": {
    "inlineMath": [['\\(', '\\)']],
    "displayMath": [["\\[", "\\]"]],
  }
}


napoleon_google_docstring = True #allows the use of Google's docstring style
napoleon_use_param = False #disables the use of :param: tags in docstrings, used to document the parameters of a function or method
napoleon_use_ivar = True #allows the use of :ivar: tags in docstrings,used to document the instance variables of a class

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = {
    '.rst': 'restructuredtext',
}

# The encoding of source files.
#
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

exclude_patterns = ['notebooks/.ipynb_checkpoints']

pygments_style = 'sphinx'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'

html_static_path = ['_static']
