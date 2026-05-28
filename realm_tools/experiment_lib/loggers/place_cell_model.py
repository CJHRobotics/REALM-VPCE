import os
import numpy as np
import h5py


class PlaceCellEnsemble:
    def __init__(self):
        self.centers = []   # (N_cells, feature_dim)
        self.radii   = []   # (N_cells,)
        self.method  = None
        self.maze    = None

    def save(self, filename, method=None, maze=None):
        """
        Save the place cell ensemble to an HDF5 file.
        Creates the directory if it does not exist.

        Structure:
            centers      (N_cells x feature_dim)  -- cluster centroids in feature space
            radii        (N_cells,)               -- RBF sigma per cell
            attrs:
                method   str   e.g. 'kmeans' or 'gmm'
                maze     str   e.g. 'LM8'
                n_cells  int
        """
        if not filename.endswith('.h5'):
            filename += '.h5'
        dir_name = os.path.dirname(filename)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with h5py.File(filename, 'w') as f:
            f.create_dataset('centers', data=np.array(self.centers), compression='gzip')
            f.create_dataset('radii',   data=np.array(self.radii))
            f.attrs['n_cells'] = len(self.radii)
            if method is not None:
                f.attrs['method'] = method
            if maze is not None:
                f.attrs['maze'] = maze

        print(f"Place cell ensemble saved to {filename}  ({len(self.radii)} cells)")

    @staticmethod
    def load(filename):
        """
        Load a place cell ensemble from an HDF5 file.
        """
        if not filename.endswith('.h5'):
            filename += '.h5'
        ensemble = PlaceCellEnsemble()
        with h5py.File(filename, 'r') as f:
            ensemble.centers = f['centers'][:]
            ensemble.radii   = f['radii'][:]
            ensemble.method  = f.attrs.get('method', None)
            ensemble.maze    = f.attrs.get('maze', None)
        return ensemble
