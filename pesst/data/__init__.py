import os
import pkg_resources


def get_data_full_name():
    resource = os.path.join("data", "LGaa.csv")
    return pkg_resources.resource_filename("pesst", resource)
