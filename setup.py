import os
from setuptools import setup
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
    "version": "1.0",  # https://packaging.python.org/guides/single-sourcing-package-version/
    "author": "Benjamin Evans and Adam Thomas",
    "author_email": "ben.d.evans@gmail.com",
    "description": ("Protein Evolution Simulation with Stability Tracking. A Python module for evolving proteins."),
    "license": "BSD",
    "keywords": "example documentation tutorial",
    "url": "http://packages.python.org/pest",
    "packages": ['pest'],
    "extras_require": {"process":  ["biopython"]},
    "long_description": read('README'),
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
