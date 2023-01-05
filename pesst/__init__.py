"""Protein Evolution Simulator with Sequence Tracking."""

__doc__ = """A Python module for evolving proteins."""
# __all__ = ["pesst"]
# __name__ = "pesst"
__project__ = "pesst"
__version__ = "1.1"

import os


def get_data_path():
    # resource = os.path.join("data", "LGaa.csv")
    # full_file_name = pkg_resources.resource_filename("pesst", resource)
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")
