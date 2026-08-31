"""
# msn

author: Antonio Gonzalez
"""
from pathlib import Path

# Directories and file names where the various components of the model
# are stored.
files = {
    'mechanisms': {
        'dir': 'mechanisms',
        'files': ''},
    'conductances': {
        'dir': 'parameters',
        'files': 'conductances.tsv'},
    'parameters': {
        'dir': 'parameters',
        'files': {
            'dmsn': 'D1_71bestFit_updRheob.pkl',
            'imsn': 'D2_34bestFit_updRheob.pkl'}},
    'morphologies': {
        'dir': 'morphologies',
        'files': {
            'dmsn': 'WT-dMSN_P270-20_1.02_SGA1-m24.swc',
            'imsn': 'WT-iMSN_P270-09_1.01_SGA2-m1.swc'}}
}


def get_paths(files):
    root = Path(__path__[0])
    paths = {}
    for key, item in files.items():
        dirname = root.joinpath(item['dir'])
        paths[key] = {}
        files = item['files']
        if type(files) is dict:
            paths[key] = {}
            for cell_type, fname in files.items():
                paths[key][cell_type] = dirname.joinpath(fname)
        else:
            paths[key] = dirname.joinpath(files)
    return paths

paths = get_paths(files)
del files, get_paths

# These imports should take place after get_paths() because they require
# file path information.
from . import cell
from . import instrumentation
from . import modulation
