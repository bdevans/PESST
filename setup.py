import os
from setuptools import setup, find_packages
import codecs

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
# def read(fname):
#     return open(os.path.join(os.path.dirname(__file__), fname)).read()


here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


setup_kwargs = {
    "name": "PESST",
    "version": "1.0.0",  # https://packaging.python.org/guides/single-sourcing-package-version/
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
    "python_requires": '>=3.6',
    "install_requires": ['numpy', 'scipy', 'pandas', 'tqdm', 'matplotlib', 'seaborn', 'imageio'],
    "extras_require": {"process":  ["biopython"]},
    "include_package_data": True,
    "long_description": read('README.md'),
    "long_description_content_type": "text/markdown",
    "classifiers": ["Development Status :: 5 - Production/Stable",
                    "Intended Audience :: Science/Research",
                    "Topic :: Scientific/Engineering",
                    "Topic :: Scientific/Engineering :: Artificial Life",
                    "Topic :: Scientific/Engineering :: Bio-Informatics",
                    "Programming Language :: Python :: 3 :: Only",
                    "Programming Language :: Python :: 3.6",
                    "Programming Language :: Python :: 3.7",
                    "License :: OSI Approved :: BSD License"],
}

setup(**setup_kwargs)
