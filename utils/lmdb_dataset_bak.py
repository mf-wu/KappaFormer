import torch
from torch.utils.data import Dataset
import lmdb
import pickle
from ase import Atoms
from torch_geometric.data import Data   
from torch_geometric.loader import DataLoader   
from torch_geometric.loader.dataloader import Collater   
from collections.abc import Mapping
from typing import Any, List, Optional, Sequence, Union

import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.typing import TensorFrame, torch_frame

class LMDBDataset(Dataset):
    def __init__(self, lmdb_path,split='train',label_keys=[]):
        lmdb_path = f'{lmdb_path}/{split}/data' 
        self.env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False)
        with self.env.begin() as txn:
            self.length = pickle.loads(txn.get(b'length'))
        print(f'read length:{self.length}')
        self.label_keys = label_keys

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            key = f"{idx}".encode("ascii")
            data = pickle.loads(txn.get(key))

        pos = torch.tensor(data['positions'], dtype=torch.float32)
        cell = torch.tensor(data['cell'], dtype=torch.float32)
        numbers =  torch.tensor(data['numbers'], dtype=torch.long) 
        natoms = len(numbers)
        
        
        data_dict = {
            "x":numbers,
            # "atoms": atoms,
            "pos": pos,
            "cell": cell,
            "atomic_numbers": numbers,
            "natoms":natoms,
             
            # "energy": energy,
            # "energy_per_atom":energy/natoms,
            # "forces": force,
            # "stress": stress,
        }
        for key in self.label_keys:
            if data.get(key):
                label_item = torch.tensor(data[key], dtype=torch.float32)
                data_dict.update({key:label_item})
            if data.get('info'):
                if data['info'].get(key):
                    label_item = torch.tensor(data['info'][key], dtype=torch.float32)
                    data_dict.update({key:label_item})
            
        return Data(**data_dict)

    @staticmethod
    def collate(batch: List[Any]) -> Any:
        elem = batch[0]
        if isinstance(elem, BaseData):
            return Batch.from_data_list(
                batch,
                follow_batch=None,
                exclude_keys=None,
            )
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, TensorFrame):
            return torch_frame.cat(batch, dim=0)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: ([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*((s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [(s) for s in zip(*batch)]

        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")

        

def get_dataloader(data_path, split, batch_size,label_keys):
    dataset = LMDBDataset(data_path,split=split,label_keys=label_keys)
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=(split=='train'),)
    return loader

def get_dataset(data_path, split, batch_size,label_keys):
    dataset = LMDBDataset(data_path,split=split,label_keys=label_keys)
    return dataset