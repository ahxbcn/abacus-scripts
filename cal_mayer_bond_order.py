"""
Calculate Mayer bond order for gamma-only calculations of ABACUS using LCAO basis.
"""
import os
from dataclasses import dataclass, field, fields
from typing import Dict, Any, List

import numpy as np


@dataclass
class AbacusNAO:
    element: str
    energy_cutoff: float
    radius: float
    Lmax: int
    l_orbs: Dict[str, int]
    orbs: List[Dict[str, int | np.ndarray]]
    mesh: np.ndarray
    dr: float
    _dynamic_fields: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __setattr__(self, name: str, value: Any) -> None:
        fixed_fields = {f.name for f in fields(self)}
        if name in fixed_fields:
            super().__setattr__(name, value)
        else:
            self._dynamic_fields[name] = value

    def __getattr__(self, name: str) -> Any:
        if name in self._dynamic_fields:
            return self._dynamic_fields[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

def read_nao_file(nao_file):
    
    with open(nao_file) as f:
        nao_file_content = f.readlines()

    summary_end_line = 0
    for linenum, line in enumerate(nao_file_content):
        if 'SUMMARY  END' in line:
            summary_end_line = linenum
            break
    
    summary = nao_file_content[1:summary_end_line-1]
    element = summary[0].split()[-1]
    energy_cutoff = float(summary[1].split()[-1])
    radius = float(summary[2].split()[-1])
    Lmax = int(summary[3].split()[-1])
    n_l_orbs = []
    for idx, line in enumerate(summary[4:]):
        n_l_orbs.append(int(line.split()[-1]))

    mesh = int(nao_file_content[summary_end_line+2].split()[-1])
    dr = float(nao_file_content[summary_end_line+3].split()[-1])
    
    orb_data_start_lines = []
    for linenum, line in enumerate(nao_file_content[summary_end_line+4:]):
        if 'Type' in line and 'L' in line and 'N' in line:
            orb_data_start_lines.append(linenum+summary_end_line+4)

    orbs = []
    for orb_idx, orb_data_start_line in enumerate(orb_data_start_lines):
        _, l, n = nao_file_content[orb_data_start_line+1].split()
        l, n = int(l), int(n)
        if orb_idx == len(orb_data_start_lines) - 1:
            orb_data_original = nao_file_content[orb_data_start_line+2:]
        else:
            orb_data_original = nao_file_content[orb_data_start_line+2:orb_data_start_lines[orb_idx+1]]

        orb_data = []
        for data_line in orb_data_original:
            data = data_line.split()
            for num in data:
                orb_data.append(float(num))

        orbs.append({'l': l, 'n': n, 'orb_data': np.array(orb_data)})


    nao = AbacusNAO(element, energy_cutoff, radius, Lmax, n_l_orbs, orbs, mesh, dr)

    return nao

def get_nao_basis_num(nao):
    """
    Get number of basis functions for a given NAO.
    """
    nbas = 0
    for i, norb in enumerate(nao.l_orbs):
        nbas += (i*2+1) * norb # Use 5D 7F basis
    
    return nbas

def read_overlap_matrix(ovlp_mat_file):
    """
    Read overlap matrix from ovlp_mat_file.
    """
    with open(ovlp_mat_file, "r") as f:
        lines = f.readlines()
    
    all_vals = []
    for line in lines:
        if not line.strip():
            continue
        vals = list(map(float, line.split()))
        all_vals.extend(vals)
    
    ndim = int(all_vals[0])
    vals = all_vals[1:]

    S = np.zeros((ndim, ndim))

    idx = 0
    for i in range(ndim):
        for j in range(i, ndim):
            S[i, j] = vals[idx]
            S[j, i] = vals[idx] # Real symmetric overlap matrix
            idx += 1
    
    return S

def read_density_matrix(rho_mat_file):
    """
    Read density matrix from rho_mat_file.
    """
    with open(rho_mat_file, "r") as f:
        lines = f.readlines()
    
    empty_lines_idx = [i for i, line in enumerate(lines) if not line.strip()]

    dim_line = empty_lines_idx[-1] - 1
    dims = [int(s) for s in lines[dim_line].split()]
    
    assert len(dims) == 2
    assert dims[0] == dims[1]

    ndim = dims[0]

    vals = []
    for iline in range(empty_lines_idx[-1]+1, len(lines)):
        line_vals = [float(x) for x in lines[iline].split()]
        vals.extend(line_vals)
    
    assert len(vals) == ndim * ndim
    dm_mat = np.array(vals).reshape(ndim, ndim)

    return dm_mat

def cal_mayer_bond_order_between_atom_pair(ovlp_mat, dm, iorb_atom1, iorb_atom2):
    """
    Calculate Mayer bond order between two atoms.
    """
    PS = dm @ ovlp_mat
    mayer_bond_order = 0
    for iorb in iorb_atom1:
        for jorb in iorb_atom2:
            mayer_bond_order += PS[iorb, jorb] * PS[jorb, iorb]
    
    return mayer_bond_order

def cal_mayer_bond_order(abacusjob_dir):
    """
    Calculate Mayer bond order from ABACUS calculation output.
    """
    from pprint import pprint
    from pathlib import Path
    from abacustest.lib_prepare.abacus import ReadInput
    from abacustest.lib_prepare.stru import AbacusSTRU

    input_params = ReadInput(os.path.join(Path(abacusjob_dir).absolute(), "INPUT"))

    assert input_params.get('nspin', 1) == 1 # Only support spin-unpolarized calculation now
    assert input_params.get('gamma_only', 1) == 1 # Only support gamma-only calculation
    assert input_params.get('out_mat_hs', 1) == 1
    assert input_params.get('out_dm', 1) == 1

    suffix = input_params.get('suffix', 'ABACUS')
    ovlp_mat_file = f"{abacusjob_dir}/OUT.{suffix}/data-0-S"
    dm_file = f"{abacusjob_dir}/OUT.{suffix}/SPIN1_DM"

    ovlp_mat = read_overlap_matrix(ovlp_mat_file)
    dm = read_density_matrix(dm_file)

    stru_file = os.path.join(Path(abacusjob_dir).absolute(), input_params.get('stru_file', 'STRU'))
    stru = AbacusSTRU.read(stru_file)
    # Read NAOs for each atoms
    naos = {}
    orb_dir = input_params.get('orbital_dir', './')
    for atom in stru.atoms:
        atom_nao = atom.orb
        if atom_nao not in naos.keys():
            nao_file = os.path.join(orb_dir, atom_nao)
            nao = read_nao_file(nao_file)
            naos[atom_nao] = nao
    
    atom_basis_nums = []
    for atom in stru.atoms:
        atom_basis_nums.append(get_nao_basis_num(naos[atom.orb]))

    print("Mayer Bond Order:")
    for i in range(stru.natoms):
        for j in range(i+1, stru.natoms):
            iorb_atom1 = [iorb for iorb in range(sum(atom_basis_nums[:i]), sum(atom_basis_nums[:i+1]))]
            iorb_atom2 = [iorb for iorb in range(sum(atom_basis_nums[:j]), sum(atom_basis_nums[:j+1]))]

            mayer_bond_order = cal_mayer_bond_order_between_atom_pair(ovlp_mat, dm, iorb_atom1, iorb_atom2)
            atomtype1, atomtype2 = stru.atoms[i].label, stru.atoms[j].label
            print(f"{atomtype1}{i} - {atomtype2}{j}: {mayer_bond_order}")

if __name__ == '__main__':
    abacusjob_dir = "/mnt/e/profsoftfiles/abacusfiles/sp/H2_out_mat_hs"
    print(f"Calculate Mayer bond order for {abacusjob_dir}")
    cal_mayer_bond_order(abacusjob_dir)

