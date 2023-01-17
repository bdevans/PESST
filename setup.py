import os
import codecs
from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    """Utility function to read the README file for the long description.

    This avoids duplication by reading the top-level README file.
    It is also easier to edit that than put a raw string below.
    """
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


setup_kwargs = {
    "name": "PESST",
    "version": "1.1.0",  # https://packaging.python.org/guides/single-sourcing-package-version/
    "author": "Benjamin Evans and Adam Thomas",
    "author_email": "ben.d.evans@gmail.com",
    "maintainer": "Benjamin Evans",
    "maintainer_email": "ben.d.evans@gmail.com",
    "description": ("Protein Evolution Simulator with Stability Tracking."),
    "license": "BSD",
    "keywords": "protein evolution simulator",
    # "url": "https://pypi.org/pesst",
    "url": "https://github.com/bdevans/PESST",
    # "packages": ['pesst'],
    "packages": find_packages(),
    "package_data": {'pesst': ['data/*.csv']},
    "python_requires": '>=3.6',
    "install_requires": [
        'numpy>1.15',
        'scipy>=1.1',
        'pandas>0.23',
        'tqdm',
        'matplotlib>=2.2',
        'seaborn',
        'imageio'],
    "extras_require": {"process":  ["biopython"]},
    "include_package_data": True,
    # "long_description": read('README.md'),
    "long_description_content_type": "text/markdown",
    "classifiers": ["Development Status :: 5 - Production/Stable",
                    "Intended Audience :: Science/Research",
                    "Topic :: Scientific/Engineering",
                    "Topic :: Scientific/Engineering :: Artificial Life",
                    "Topic :: Scientific/Engineering :: Bio-Informatics",
                    "Programming Language :: Python :: 3 :: Only",
                    "Programming Language :: Python :: 3.7",
                    "Programming Language :: Python :: 3.8",
                    "Programming Language :: Python :: 3.9",
                    "Programming Language :: Python :: 3.10",
                    "Programming Language :: Python :: 3.11",
                    "License :: OSI Approved :: BSD License"],
}

setup(**setup_kwargs)
