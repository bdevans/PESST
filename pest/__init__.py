"""Protein Evolution with Sequence Tracking."""

__doc__ = """A Python module for evolving proteins."""
# Main module file for PyRhO

__all__ = ["pest"]
__project__ = "pest"
__version__ = "1.0"

import os


def get_data_path():
    # resource = os.path.join("data", "LGaa.csv")
    # full_file_name = pkg_resources.resource_filename("pest", resource)
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")
