"""dataset.py"""

import h5py
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

class H5GraphDataset(Dataset):
    def __init__(self, dInfo, dset_path, length=0, short=False, portion=0.5):
        super().__init__()
        self.dInfo = dInfo
        self.dset_path = dset_path

        self.z_dim = len(dInfo['dataset']['state_variables'])
        self.q_dim = 2 if dInfo['dataset']['dataset_dim'] == '2D' else 3
        self.samplingFactor = dInfo['dataset']['samplingFactor']
        self.dt = dInfo['dataset']['dt'] * self.samplingFactor
        self.dims = {'z': self.z_dim, 'q': 0, 'q_0': self.q_dim, 'n': 9, 'f': 0, 'g': 0}

        with h5py.File(self.dset_path, "r") as hf:
            self.sample_keys = list(hf.keys())
            if "pairs" in hf:
                self.is_test = True
                self.mapping = [(None, pk) for pk in sorted(hf["pairs"].keys(), key=lambda k: int(k.split('_')[1]))]
            else:
                self.is_test = False
                self.mapping = []
                for sk in self.sample_keys:
                    pair_keys = sorted(hf[sk]["pairs"].keys(), key=lambda k: int(k.split('_')[1]))
                    self.mapping.extend([(sk, pk) for pk in pair_keys])

        if length > 0:
            self.mapping = self.mapping[:length]
        if short:
            self.mapping = self.mapping[:int(len(self.mapping) * portion)]

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        """Load data from the HDF5 file and return a Pytorch Geometric (PyG) 'Data' object"""
        with h5py.File(self.dset_path, "r") as hf:
            if self.is_test:
                edge_index = torch.tensor(hf["edge_index"][...], dtype=torch.long)
                edge_attr = torch.tensor(hf["edge_attr"][...], dtype=torch.float32)
                pos = torch.tensor(hf["pos"][...], dtype=torch.float32)
                node_type = torch.tensor(hf["node_type"][...], dtype=torch.long).squeeze()
                elements = torch.tensor(hf["elements"][...], dtype=torch.long).T
                dt = torch.tensor(hf.attrs["dt"], dtype=torch.float32)

                _, pair_key = self.mapping[idx]
                pair_grp = hf["pairs"][pair_key]
            else:
                sample_key, pair_key = self.mapping[idx]
                sample_grp = hf[sample_key]

                edge_index = torch.tensor(sample_grp["edge_index"][...], dtype=torch.long)
                edge_attr = torch.tensor(sample_grp["edge_attr"][...], dtype=torch.float32)
                pos = torch.tensor(sample_grp["pos"][...], dtype=torch.float32)
                node_type = torch.tensor(sample_grp["node_type"][...], dtype=torch.long).squeeze()
                elements = torch.tensor(sample_grp["elements"][...], dtype=torch.long).T
                dt = torch.tensor(sample_grp.attrs["dt"], dtype=torch.float32)

                pair_grp = sample_grp["pairs"][pair_key]

            x = torch.tensor(pair_grp["x"][...], dtype=torch.float32)
            y = torch.tensor(pair_grp["y"][...], dtype=torch.float32)

        return Data(
            x=x,
            y=y,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            n=node_type,
            elements=elements,
            dt=dt
        )

if __name__ == '__main__':
    pass
