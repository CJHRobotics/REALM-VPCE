import os
import numpy as np
import h5py


class PovDataset:
    def __init__(self):
        self.multimodal_features = []
        self.cnn_features = []
        self.x = []
        self.y = []
        self.theta = []

    def add_observations(self, multimodal_feature_vector, cnn_feature_vector, x, y, theta):
        """
        Adds a new datapoint to the dataset.
        """
        self.multimodal_features.append(multimodal_feature_vector)
        self.cnn_features.append(cnn_feature_vector)
        self.x.append(x)
        self.y.append(y)
        self.theta.append(theta)

    def save_dataset(self, filename):
        """
        Save the dataset to an HDF5 file.
        Creates the directory if it does not exist.

        Structure:
            multimodal_features  (N x feature_dim)
            cnn_features         (N x cnn_dim)  -- omitted if not collected
            x                    (N,)
            y                    (N,)
            theta                (N,)
        """
        if not filename.endswith('.h5'):
            filename += '.h5'
        dir_name = os.path.dirname(filename)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with h5py.File(filename, 'w') as f:
            f.create_dataset('multimodal_features',
                             data=np.array(self.multimodal_features),
                             compression='gzip')
            # Only write CNN features if they were actually collected
            if self.cnn_features and self.cnn_features[0] is not None:
                f.create_dataset('cnn_features',
                                 data=np.array(self.cnn_features),
                                 compression='gzip')
            f.create_dataset('x',     data=np.array(self.x))
            f.create_dataset('y',     data=np.array(self.y))
            f.create_dataset('theta', data=np.array(self.theta))

        print(f"Dataset saved to {filename}  ({len(self.x)} observations)")

    @staticmethod
    def load_dataset(filename):
        """
        Load a dataset from an HDF5 file.
        """
        if not filename.endswith('.h5'):
            filename += '.h5'
        dataset = PovDataset()
        with h5py.File(filename, 'r') as f:
            dataset.multimodal_features = list(f['multimodal_features'][:])
            dataset.cnn_features = list(f['cnn_features'][:]) if 'cnn_features' in f else []
            dataset.x     = list(f['x'][:])
            dataset.y     = list(f['y'][:])
            dataset.theta = list(f['theta'][:])
        return dataset
