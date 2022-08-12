import importlib


def is_neptune_available():
    return importlib.util.find_spec("neptune") is not None


def is_layer_available():
    return importlib.util.find_spec("layer") is not None


def check_requirements(package_names):
    """
    Raise error if module is not installed.
    """
    missing_packages = []
    for package_name in package_names:
        if importlib.util.find_spec(package_name) is None:
            missing_packages.append(package_name)
    if missing_packages:
        raise ImportError(f"The following packages are required to use this module: {missing_packages}")
    yield
