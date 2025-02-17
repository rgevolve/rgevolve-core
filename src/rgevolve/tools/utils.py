import importlib
import appdirs
import os
import h5py
import numpy as np
import re
from functools import lru_cache

def get_data_path(package_name):
    """Return the path to the data file."""
    try:
        return importlib.resources.files(package_name).joinpath('data.h5')
    except AttributeError:
        import pkg_resources
        return pkg_resources.resource_filename(package_name, "data.h5")

def get_cache_path(package_name):
    """Return the path to the cache file."""
    cachedir = appdirs.user_data_dir('rgevolve')
    if not os.path.exists(cachedir):
        os.makedirs(cachedir)
    return os.path.join(cachedir, f'{package_name}.h5')

def update_cache(cache_path, package_name, data_h5):
    if os.path.exists(cache_path):
        mode = "r+"  # Open in read/write mode if it exists
    else:
        mode = "w"   # Create a new file if it doesn't exist
    try:
        with h5py.File(cache_path, mode) as h5file:
            if mode == "r+":
                for sector, RG_evolution in data_h5['RG evolution'].items():
                    if sector in h5file and h5file[sector].attrs['hash'] == hash(RG_evolution):
                        continue
                    elif sector in h5file:
                        print(f"Updating cache file for {package_name}")
                        print(f"Updating sector {sector} (hash mismatch)")
                        inverses = np.array([
                            np.linalg.inv(matrix)
                            for matrix in RG_evolution
                        ])
                        h5file[sector][...] = inverses
                        h5file[sector].attrs['hash'] = hash(RG_evolution)
                    else:
                        print(f"Updating cache file for {package_name}")
                        print(f"Adding sector {sector}")
                        inverses = np.array([
                            np.linalg.inv(matrix)
                            for matrix in RG_evolution
                        ])
                        h5file.create_dataset(sector, data=inverses, compression="gzip")
                        h5file[sector].attrs['hash'] = hash(RG_evolution)
            else:
                print(f"Creating new cache file for {package_name}")
                for sector, RG_evolution in data_h5['RG evolution'].items():
                    print(f"Adding sector {sector}")
                    inverses = np.array([
                        np.linalg.inv(matrix)
                        for matrix in RG_evolution
                    ])
                    h5file.create_dataset(sector, data=inverses, compression="gzip")
                    h5file[sector].attrs['hash'] = hash(RG_evolution)
    except BlockingIOError:
        pass

def load_data(package_name):
    data_path = get_data_path(package_name)
    cache_path = get_cache_path(package_name)
    data_h5 = h5py.File(data_path, 'r')
    update_cache(cache_path, package_name, data_h5)
    cache_h5 = h5py.File(cache_path, 'r')
    evolution = {
        'regular': data_h5['RG evolution'],
        'inverse': cache_h5
    }
    translation = data_h5['Translation']
    return evolution, translation

def normalize(name):
    return re.sub(r'[^a-zA-Z0-9]+', '-', name).strip('-').lower()

@lru_cache(maxsize=None)
def get_module(eft, basis):
    module_name = f"rgevolve.{normalize(eft)}.{normalize(basis)}"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ImportError(
            f"The module '{module_name}' is not installed. If available, install it with:\n"
            f"    pip install {module_name}"
        )

@lru_cache(maxsize=None)
def evolution_data(eft, basis):
    module = get_module(eft, basis)
    return module.evolution

@lru_cache(maxsize=None)
def translation_data(eft, basis):
    module = get_module(eft, basis)
    return module.translation
