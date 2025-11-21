# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'Distribution Shift Lane Perception'
copyright = '2025 - Hafeez Kahn & Alexander Doyle'
author = 'Hafeez Kahn & Alexander Doyle'

release = '0.0'
version = '0.1.0'

# -- General configuration

autodoc_mock_imports = [
    # The core math/data libraries
    "numpy",
    "scipy",
    "sympy",
    "h5py",
    "networkx",

    # The Deep Learning frameworks (The heaviest ones)
    "torch",
    "torchvision",
    "torch_two_sample",  # Your custom git dependency
    "triton",            # GPU optimization library (often fails on CPUs)
    
    # TensorFlow stack (You have this installed, so we mock it)
    "tensorflow",
    "tensorboard",
    "keras",

    # Image & Utilities
    "PIL",   # This mocks 'Pillow'
    "tqdm",  # Progress bars
    
    # System/Hardware specific (Safest to mock)
    "nvidia",
    "cuda",
]

autodoc_mock_imports = ["torch_two_sample", "torch", "numpy"]

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
