import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from models.kappaformer.kappaformer import Kappaformer
from datasets.lmdb_dataset import get_dataloader
import os
from ase import Atoms
from models.kappaformer.calculator import BasicCalculator
from torch_geometric.data import Data
from ase.io import read, write
from torch_geometric.loader.dataloader import DataLoader
from ase.calculators.calculator import Calculator
from typing import Any, Dict, List, Optional, Tuple


class InferCalculator(Calculator):
    implemented_properties = ["B_VRH", "G_VRH", "log_kappa"]
    def __init__(
        self,
        model: Kappaformer,
        load_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model.to(device)
        self.device = device

        state_dict= torch.load(load_path, map_location=device) # ['state_dict']
        self.model.load_state_dict(state_dict,strict=True)
        # Important
        self.model.eval()
        
    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: Optional[list] = None,
        system_changes: Optional[list] = None,
    ):
        if atoms == None:
            atoms = self.atoms
        super().calculate(
            atoms=atoms, properties=properties, system_changes=system_changes
        )
        # Basic attributes
        attribute_calc = BasicCalculator(atoms)
        density_val = attribute_calc.mass_density()
        n_atoms_val = attribute_calc.natoms()
        volume_val = attribute_calc.volume
        ma_val = attribute_calc.Ma()

        mp_id = atoms.info["material_id"]
        material = atoms.info["formula_pretty"]
        nsites = atoms.info["nsites"]
        Ehull = atoms.info["energy_above_hull"]
        dim = atoms.info["dim"]
        # crystal structure
        pos =  torch.tensor(atoms.get_positions()).float()
        atomic_numbers = torch.tensor(atoms.get_atomic_numbers())
        cell = torch.tensor(atoms.get_cell()).float()
        natoms = torch.tensor([len(atoms)])
        
        sample = Data(x = atomic_numbers,
                        atomic_numbers=atomic_numbers,
                        pos = pos,
                        natoms=natoms,
                        cell = cell.unsqueeze(0),
                        pbc = torch.BoolTensor([True,True,True]),
                        density = density_val,
                        n_atoms = n_atoms_val,
                        V1 = volume_val,
                        Ma = ma_val,
                        mp_id = mp_id,
                        material = material,
                        nsites = nsites,
                        Ehull = Ehull,
                        dim = dim
                        )
        
        data_loader = DataLoader([sample],batch_size=1)
        data_batch = next(iter(data_loader))
        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor):
                data_batch[k] = v.to(self.device)

        mp = data_batch['mp_id']
        if isinstance(mp, (list, tuple)) and len(mp) == 1:
            mp = mp[0]
        if isinstance(mp, torch.Tensor):
            mp = mp.detach().cpu().item() if mp.dim() == 0 else mp.detach().cpu().numpy().tolist()
        
        mat = data_batch['material']
        if isinstance(mat, (list, tuple)) and len(mat) == 1:
            mat = mat[0]

        nsites = data_batch['nsites']
        if isinstance(nsites, (list, tuple)) and len(nsites) == 1:
            nsites = nsites[0]
        if isinstance(nsites, torch.Tensor):
            nsites = nsites.detach().cpu().item()
        
        Ehull = data_batch['Ehull']
        if isinstance(Ehull, (list, tuple)) and len(Ehull) == 1:
            Ehull = Ehull[0]
        if isinstance(Ehull, torch.Tensor):
            Ehull = Ehull.detach().cpu().item()
        
        dim = data_batch['dim']
        if isinstance(dim, (list, tuple)) and len(dim) == 1:
            dim = dim[0]
        if isinstance(dim, torch.Tensor):
            dim = dim.detach().cpu().item()

        # model inference
        b, g, kappa_log, harm_embedding, anharm_embedding = self.model(data_batch)
        out = mp, mat, nsites, Ehull, dim, b, g, kappa_log, harm_embedding, anharm_embedding
        return out
        

def main(args):
    atoms_list = read(args.input, index=":")
    print("Number of structures:", len(atoms_list))
    calc = InferCalculator(Kappaformer(enable_kappa=True, return_embedding=True), load_path=args.ckpt_path)
    mp_ids, materials, nsitess, Ehulls, dims, b_preds, g_preds, kappa_preds, harm_embeddings, anharm_embeddings = [], [], [], [], [], [], [], [], [], []
    for atoms in tqdm(atoms_list):
        with torch.no_grad():
            mp, mat, nsites, Ehull, dim, b, g, kappa_log, harm_embedding, anharm_embedding = calc.calculate(atoms)
            kappa = 10 ** kappa_log
        mp_ids.append(mp)  
        materials.append(mat)
        nsitess.append(nsites)
        Ehulls.append(Ehull)
        dims.append(dim)
        b_preds.append(float(b))
        g_preds.append(float(g))
        kappa_preds.append(float(kappa))
        harm_embeddings.append(harm_embedding.detach())
        anharm_embeddings.append(anharm_embedding.detach())
        del b, g, kappa_log, kappa, harm_embedding, anharm_embedding
        torch.cuda.empty_cache()
    
    b_ps = [x.item() if isinstance(x, torch.Tensor) else float(x) for x in b_preds]
    g_ps = [x.item() if isinstance(x, torch.Tensor) else float(x) for x in g_preds]
    kappa_ps = [x.item() if isinstance(x, torch.Tensor) else float(x) for x in kappa_preds]
    df_kappa = pd.DataFrame({
        "mp_id": mp_ids,
        "material": materials,
        "nsites": nsitess,
        "Ehull": Ehulls,
        "dim": dims,
        "b_pred": b_ps,
        "g_pred": g_ps,
        "kappa_pred": kappa_ps
    })
    # Embeddings
    harm_ebeds = np.vstack([e.cpu().numpy() for e in harm_embeddings])
    anharm_ebeds = np.vstack([e.cpu().numpy() for e in anharm_embeddings])
    df_harm_ebeds = pd.DataFrame(harm_ebeds, columns=[f"Harm_emb_{i}" for i in range(harm_ebeds.shape[1])])
    df_anharm_ebeds = pd.DataFrame(anharm_ebeds, columns=[f"Anharm_emb_{i}" for i in range(anharm_ebeds.shape[1])])
    # Concatenate predictions and embeddings
    df = pd.concat([df_kappa, df_harm_ebeds, df_anharm_ebeds], axis=1)

    original_path = args.output_path
    dir_path = os.path.dirname(original_path)
    file_name = os.path.basename(original_path)  # preds_embeddings.csv
    name, ext = os.path.splitext(file_name)  # name='preds_embeddings', ext='.csv'名
    file = os.path.join(dir_path, f"{name}{ext}")

    df.to_csv(file, index=False)
    print(f"Results saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract predictions and embeddings")

    parser.add_argument("--input", type=str, required=True,
                        help="Path to dataset: Input .xyz file")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the output CSV")

    args = parser.parse_args()
    main(args)