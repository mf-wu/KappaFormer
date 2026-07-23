import os
os.chdir('/share/home/u15502/mfwu/kappaformer')
from ase.calculators.calculator import Calculator
from ase.data import atomic_masses
from models.kappaformer.kappaformer  import Kappaformer
from typing import Dict, List, Optional, Tuple
from ase import Atoms
import torch
from torch_geometric.loader.dataloader import DataLoader
from torch_geometric.data import Data

# class AttributeCalculator:
#     def __init__(self, atoms):
#         self.atoms = atoms
#         self.symbols = atoms.get_chemical_symbols()
#         self.atomic_numbers = atoms.get_atomic_numbers()
#         self.unique_elements = sorted(set(self.symbols), key=self.symbols.index)
#         self.natoms_list = [self.symbols.count(sym) for sym in self.unique_elements]

#         self.atomic_weight_list = [
#             atomic_masses[self.atomic_numbers[self.symbols.index(sym)]]
#             for sym in self.unique_elements
#         ]
#         self.volume = atoms.get_volume()
    
#     def molecular_weight(self):
#         #atomic_weight_list in  atomic mass units (u)
#         # 1 u = 1 g/mol/(N_A) = 1.0/(6.02214085774e+23) g = 1.6605390402231174e-24 g
#         mol_weight=0.0
#         for i in range(len(self.natoms_list)):
#             mol_weight += self.natoms_list[i] * self.atomic_weight_list[i]

#         #molecular weight in unit of atomic mass units (u)
#         return mol_weight

#     def mass(self):
#         #atomic_weight_list in  atomic mass units (u)
#         # 1 u = 1 g/mol/(N_A)
#         #  Avogadro's number, in unit of mol^{-1}.
#         NA = 6.02214085774e+23
#         molw = self.molecular_weight()
#         print('Average atomic mass [g/mol]:{:12.5f}'.format(molw/sum(self.natoms_list)))
#         mass_in_kg = (molw/NA)*0.001
#         #print 'Average atomic mass [kg/atom]:', mass_in_kg/sum(natoms_list)
#         #print 'Total mass [kg]:', mass_in_kg
#         return mass_in_kg

#     def mass_density(self):
#         # volume in unit of angstrom^{-3}.
#         #atomic_weight_list in  atomic mass units (u)
#         # 1 u = 1 g/mol/(N_A)
#         # density in unit of kg/m^3.
#         mass_in_kg = self.mass()
#         rho = mass_in_kg/(self.volume*1.0e-30)
#         print('Density [kg/m^3]:{:12.5f}'.format(rho))
#         return rho
class BasicCalculator:
    def __init__(self, atoms):
        self.atoms = atoms
        self.symbols = atoms.get_chemical_symbols()
        self.atomic_numbers = atoms.get_atomic_numbers()
        self.unique_elements = sorted(set(self.symbols), key=self.symbols.index)
        self.natoms_list = [self.symbols.count(sym) for sym in self.unique_elements]

        self.atomic_weight_list = [
            atomic_masses[self.atomic_numbers[self.symbols.index(sym)]]
            for sym in self.unique_elements
        ]
        self.volume = atoms.get_volume()

    def molecular_weight(self):
        #atomic_weight_list in  atomic mass units (u)
        # 1 u = 1 g/mol/(N_A) = 1.0/(6.02214085774e+23) g = 1.6605390402231174e-24 g
        mol_weight=0.0
        for i in range(len(self.natoms_list)):
            mol_weight += self.natoms_list[i] * self.atomic_weight_list[i]

        #molecular weight in unit of atomic mass units (u)
        return mol_weight

    def mass(self):
        #atomic_weight_list in  atomic mass units (u)
        # 1 u = 1 g/mol/(N_A)
        #  Avogadro's number, in unit of mol^{-1}.
        NA = 6.02214085774e+23
        molw = self.molecular_weight()
        print('Average atomic mass [g/mol]:{:12.5f}'.format(molw/sum(self.natoms_list)))
        mass_in_kg = (molw/NA)*0.001
        #print 'Average atomic mass [kg/atom]:', mass_in_kg/sum(natoms_list)
        #print 'Total mass [kg]:', mass_in_kg
        return mass_in_kg

    def mass_density(self):
        # volume in unit of angstrom^{-3}.
        #atomic_weight_list in  atomic mass units (u)
        # 1 u = 1 g/mol/(N_A)
        # density in unit of kg/m^3.
        mass_in_kg = self.mass()
        rho = mass_in_kg/(self.volume*1.0e-30)
        print('Density [kg/m^3]:{:12.5f}'.format(rho))
        return rho
    
    # def get_change_volume(self):
    #     structure = AseAtomsAdaptor.get_structure(self.atoms)
    #     structure.apply_strain(-0.01)
    #     atoms_change = AseAtomsAdaptor.get_atoms(structure)
    #     volume_2 = atoms_change.get_volume()
    #     return volume_2
    
    def natoms(self):
        return sum(self.natoms_list)
    
    def Ma(self):
        return self.molecular_weight()/float(self.natoms())
    

class KappaFormerCalculator(Calculator):
    implemented_properties = ["B_VRH", "G_VRH", "log_kappa"]
    def __init__(
        self,
        model: Kappaformer,
        load_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.device = device
    
        state_dict= torch.load(load_path, map_location=device) # ['state_dict']
        self.model.load_state_dict(state_dict,strict=True)
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
                        Ma = ma_val
                        )
        
        data_loader = DataLoader([sample],batch_size=1)
        data_batch = next(iter(data_loader))
        
        result = self.model(data_batch)
        if len(result) == 1:
            self.results.update(direct_log_kappa = result.item() )
        elif len(result) == 2:
            self.results.update(B_VRH = result[0].item() )
            self.results.update(G_VRH = result[1].item() )
        elif len(result) == 3:
            self.results.update(B_VRH = result[0].item() )
            self.results.update(G_VRH = result[1].item() )
            self.results.update(log_kappa = result[2].item() )