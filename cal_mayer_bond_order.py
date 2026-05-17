"""
Calculate Mayer bond order for gamma-only calculations of ABACUS using LCAO basis.
"""
import os
from dataclasses import dataclass, field, fields
from typing import Dict, Any, List
import argparse

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

    mesh = int(nao_file_content[summary_end_line + 2].split()[-1])
    dr = float(nao_file_content[summary_end_line + 3].split()[-1])

    orb_data_start_lines = []
    for linenum, line in enumerate(nao_file_content[summary_end_line + 4 :]):
        if "Type" in line and "L" in line and "N" in line:
            orb_data_start_lines.append(linenum + summary_end_line + 4)

    orbs = []
    for orb_idx, orb_data_start_line in enumerate(orb_data_start_lines):
        _, l, n = nao_file_content[orb_data_start_line + 1].split()
        l, n = int(l), int(n)
        if orb_idx == len(orb_data_start_lines) - 1:
            orb_data_original = nao_file_content[orb_data_start_line + 2 :]
        else:
            orb_data_original = nao_file_content[orb_data_start_line+2:orb_data_start_lines[orb_idx+1]]

        orb_data = []
        for data_line in orb_data_original:
            data = data_line.split()
            for num in data:
                orb_data.append(float(num))

        orbs.append({"l": l, "n": n, "orb_data": np.array(orb_data)})

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
    Supports both real and complex (real,imag) formats.
    """
    with open(ovlp_mat_file, "r") as f:
        lines = f.readlines()

    first_line = lines[0].strip()
    is_complex = "(" in first_line and "," in first_line and ")" in first_line

    all_nums = []
    for line in lines:
        if not line.strip():
            continue
        if is_complex:
            line = line.replace("(", " ").replace(")", " ").replace(",", " ")
        all_nums.extend(list(map(float, line.split())))

    if is_complex:
        ndim = int(all_nums[0])
        complex_vals = []
        for i in range(1, len(all_nums), 2):
            complex_vals.append(complex(all_nums[i], all_nums[i + 1]))

        S = np.zeros((ndim, ndim), dtype=np.complex128)
        idx = 0
        for i in range(ndim):
            for j in range(i, ndim):
                S[i, j] = complex_vals[idx]
                S[j, i] = complex_vals[idx].conjugate()
                idx += 1
    else:
        ndim = int(all_nums[0])
        vals = all_nums[1:]

        S = np.zeros((ndim, ndim))
        idx = 0
        for i in range(ndim):
            for j in range(i, ndim):
                S[i, j] = vals[idx]
                S[j, i] = vals[idx]  # Real symmetric overlap matrix
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

def cal_mayer_bond_order_between_atom_pair(iorb_atom1, iorb_atom2, ovlp_mat, dm, dm_dn=None):
    """
    Calculate Mayer bond order between two atoms.
    """
    mayer_bond_order = 0
    if dm_dn is None:
        PS = dm @ ovlp_mat
        for iorb in iorb_atom1:
            for jorb in iorb_atom2:
                mayer_bond_order += (PS[iorb, jorb] * PS[jorb, iorb]).real
    else:
        PS = dm @ ovlp_mat
        PS_dn = dm_dn @ ovlp_mat
        for iorb in iorb_atom1:
            for jorb in iorb_atom2:
                mayer_bond_order += (PS[iorb, jorb] * PS[jorb, iorb] + PS_dn[iorb, jorb] * PS_dn[jorb, iorb]).real
        mayer_bond_order *= 2

    return mayer_bond_order


def read_wfc_nao_k(file_path):
    """Parse WFC_NAO_K*.txt wavefunction file."""
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    nbands = None
    nlocal = None
    for line in lines:
        if line.endswith("(number of bands)"):
            nbands = int(line.split()[0])
        elif line.endswith("(number of orbitals)"):
            nlocal = int(line.split()[0])

    if nbands is None or nlocal is None:
        raise ValueError(f"Failed to parse {file_path}")

    wfc = np.zeros((nlocal, nbands), dtype=np.complex128)
    wg = np.zeros(nbands)
    ib = 0
    ilocal = 0
    reading_band = False

    for line in lines:
        if line.endswith("(band)"):
            ib = int(line.split()[0]) - 1
            ilocal = 0
            reading_band = True
        elif line.endswith("(Ry)"):
            continue
        elif line.endswith("(Occupations)"):
            wg[ib] = float(line.split()[0])
        elif reading_band and not line.endswith(")"):
            nums = line.split()
            try:
                c = [complex(float(nums[i]), float(nums[i + 1]))
                     for i in range(0, len(nums), 2)]
            except Exception:
                continue
            for ci in c:
                if ilocal < nlocal:
                    wfc[ilocal, ib] = ci
                    ilocal += 1

    return wfc, wg


def calculate_density_matrix_k(wfc, wg):
    """Build density matrix P_k = C_k * diag(wg_k) * C_k^H."""
    wfc_weighted = wfc * np.sqrt(wg)[np.newaxis, :]
    return wfc_weighted @ wfc_weighted.conj().T


def cal_mayer_bond_order_between_atom_pair_k(iorb_atom1, iorb_atom2, ovlp_mat, dm):
    """Mayer bond order between two atoms at a single k-point."""
    mayer_bond_order_k = 0
    PS = dm @ ovlp_mat
    for iorb in iorb_atom1:
        for jorb in iorb_atom2:
            mayer_bond_order_k += (PS[iorb, jorb] * PS[jorb, iorb]).real
    return mayer_bond_order_k


def read_kpoint_weights(kpoints_file):
    """Parse IBZ k-point weights from the kpoints file."""
    with open(kpoints_file, "r") as f:
        lines = f.readlines()

    weights = []
    in_table = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_table:
                break
            continue
        if "KPOINTS" in stripped and "DIRECT_X" in stripped:
            in_table = True
            continue
        if in_table:
            parts = stripped.split()
            if len(parts) >= 5 and parts[0].isdigit():
                weights.append(float(parts[4]))

    if not weights:
        raise ValueError(f"No k-point weights found in {kpoints_file}")
    return weights


def cal_mayer_bond_order(abacusjob_dir):
    """
    Calculate Mayer bond order from ABACUS calculation output.
    """
    import glob
    from pathlib import Path
    from abacustest.lib_prepare.abacus import ReadInput
    from abacustest.lib_prepare.stru import AbacusSTRU

    input_params = ReadInput(os.path.join(Path(abacusjob_dir).absolute(), "INPUT"))
    nspin = input_params.get("nspin", 1)
    gamma_only = input_params.get("gamma_only")

    out_mat_hs = input_params.get("out_mat_hs", 1)
    out_mat_hs = out_mat_hs[0] if isinstance(out_mat_hs, list) else out_mat_hs
    assert out_mat_hs == 1, f"out_mat_hs must be 1, got {out_mat_hs}"

    suffix = input_params.get('suffix', 'ABACUS')
    out_dir = f"{abacusjob_dir}/OUT.{suffix}"

    stru_file = os.path.join(Path(abacusjob_dir).absolute(), input_params.get('stru_file', 'STRU'))
    stru = AbacusSTRU.read(stru_file)
    naos = {}
    orb_dir = input_params.get('orbital_dir') or str(Path(abacusjob_dir).absolute())
    for atom in stru.atoms:
        atom_nao = atom.orb
        if atom_nao not in naos.keys():
            nao_file = os.path.join(orb_dir, atom_nao)
            nao = read_nao_file(nao_file)
            naos[atom_nao] = nao

    atom_basis_nums = []
    for atom in stru.atoms:
        atom_basis_nums.append(get_nao_basis_num(naos[atom.orb]))

    atom_orb_ranges = []
    start = 0
    for n in atom_basis_nums:
        end = start + n
        atom_orb_ranges.append((start, end))
        start = end

    if gamma_only:
        assert input_params.get("out_dm", 1) == 1
        ovlp_mat = read_overlap_matrix(f"{out_dir}/data-0-S")
        dm = read_density_matrix(f"{out_dir}/SPIN1_DM")
        if nspin == 2:
            dm_dn = read_density_matrix(f"{out_dir}/SPIN2_DM")
        else:
            dm_dn = None

        print("Mayer Bond Order (gamma-only):")
        for i in range(stru.natoms):
            for j in range(i + 1, stru.natoms):
                s1, e1 = atom_orb_ranges[i]
                s2, e2 = atom_orb_ranges[j]
                mayer_bond_order = cal_mayer_bond_order_between_atom_pair(
                    list(range(s1, e1)), list(range(s2, e2)), ovlp_mat, dm, dm_dn)
                if mayer_bond_order > 0.2:
                    print(f"  {stru.atoms[i].label}{i + 1} - {stru.atoms[j].label}{j + 1}: {mayer_bond_order:.6f}")
    else:
        wfc_files = sorted(glob.glob(os.path.join(out_dir, "WFC_NAO_K*.txt")))
        if not wfc_files:
            raise FileNotFoundError(f"No WFC_NAO_K*.txt files found in {out_dir}")

        nwfc = len(wfc_files)
        kpoints_file = os.path.join(out_dir, "kpoints")
        wk = read_kpoint_weights(kpoints_file)
        nk = len(wk)

        is_spin2_multik = (nspin == 2 and nwfc == 2 * nk)
        if not is_spin2_multik and nwfc != nk:
            raise ValueError(
                f"Number of k-point weights ({nk}) does not match "
                f"number of WFC files ({nwfc})"
            )

        if is_spin2_multik:
            print(f"Mayer Bond Order (multi-k, nspin=2, {nk} k-points):")
        else:
            print(f"Mayer Bond Order (multi-k, {nk} k-points):")

        total_bond_order = {}
        for i in range(stru.natoms):
            for j in range(i + 1, stru.natoms):
                total_bond_order[(i, j)] = 0.0

        if is_spin2_multik:
            for ik in range(nk):
                wfc_up, wg_up = read_wfc_nao_k(
                    os.path.join(out_dir, f"WFC_NAO_K{ik + 1}.txt"))
                dm_up = calculate_density_matrix_k(wfc_up, wg_up)

                wfc_dn, wg_dn = read_wfc_nao_k(
                    os.path.join(out_dir, f"WFC_NAO_K{ik + 1 + nk}.txt"))
                dm_dn = calculate_density_matrix_k(wfc_dn, wg_dn)

                ovlp_file = os.path.join(out_dir, f"data-{ik}-S")
                ovlp_mat = read_overlap_matrix(ovlp_file)

                for i in range(stru.natoms):
                    for j in range(i + 1, stru.natoms):
                        s1, e1 = atom_orb_ranges[i]
                        s2, e2 = atom_orb_ranges[j]
                        bo_up = cal_mayer_bond_order_between_atom_pair_k(
                            list(range(s1, e1)), list(range(s2, e2)), ovlp_mat, dm_up)
                        bo_dn = cal_mayer_bond_order_between_atom_pair_k(
                            list(range(s1, e1)), list(range(s2, e2)), ovlp_mat, dm_dn)
                        total_bond_order[(i, j)] += 2.0 * (bo_up + bo_dn) / wk[ik]
        else:
            for ik in range(nk):
                wfc, wg = read_wfc_nao_k(
                    os.path.join(out_dir, f"WFC_NAO_K{ik + 1}.txt"))
                dm_k = calculate_density_matrix_k(wfc, wg)

                ovlp_file = os.path.join(out_dir, f"data-{ik}-S")
                ovlp_mat = read_overlap_matrix(ovlp_file)

                for i in range(stru.natoms):
                    for j in range(i + 1, stru.natoms):
                        s1, e1 = atom_orb_ranges[i]
                        s2, e2 = atom_orb_ranges[j]
                        bo_k = cal_mayer_bond_order_between_atom_pair_k(
                            list(range(s1, e1)), list(range(s2, e2)), ovlp_mat, dm_k)
                        total_bond_order[(i, j)] += bo_k / wk[ik]

        for i in range(stru.natoms):
            for j in range(i + 1, stru.natoms):
                mayer_bond_order = total_bond_order[(i, j)]
                if mayer_bond_order > 0.2:
                    print(f"  {stru.atoms[i].label}{i + 1} - {stru.atoms[j].label}{j + 1}: {mayer_bond_order:.6f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--abacusjob_dir", type=str, default="./", help="ABACUS job directory to calculate Mayer bond order")
    args = parser.parse_args()

    abacusjob_dir = args.abacusjob_dir
    print(f"Calculate Mayer bond order for {abacusjob_dir}")
    cal_mayer_bond_order(abacusjob_dir)
