"""
Calculate Mayer bond order from real-space (R-space) density matrix and overlap matrix.
Uses ABACUS output files: data-DMR-sparse_SPIN*.csr and data-SR-sparse_SPIN*.csr.

Usage:
    python cal_mayer_bond_order_r.py -j /path/to/job
    python cal_mayer_bond_order_r.py -j /path/to/job --cutoff 2.0
    python cal_mayer_bond_order_r.py -j /path/to/job --pairs "1-2,1-3"
"""
import os
import argparse

import numpy as np


# ---- CSR file parsing ----

def parse_csr_header(lines):
    """Parse header lines of a CSR matrix file.
    Returns (nbasis, nR) and the line index where data starts.
    """
    nbasis = None
    nR = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("Matrix Dimension"):
            nbasis = int(s.split(":")[1].strip())
        elif s.startswith("Matrix number"):
            nR = int(s.split(":")[1].strip())
        if nbasis is not None and nR is not None:
            return nbasis, nR, i + 1
    raise ValueError("Could not parse CSR header")


def parse_csr_matrix(lines, start_line, nbasis):
    """Parse a single CSR (rx, ry, rz, nnz, values, col_idx, row_ptr) block.
    Returns (rx, ry, rz, csr_data) where csr_data = (values, col_ind, row_ptr).
    """
    if start_line >= len(lines):
        return None

    parts = lines[start_line].strip().split()
    if len(parts) < 4:
        return None

    rx = int(parts[0])
    ry = int(parts[1])
    rz = int(parts[2])
    nnz = int(parts[3])

    # Read values line
    vals = [float(x) for x in lines[start_line + 1].strip().split()]
    # Read column indices line
    col_ind = [int(x) for x in lines[start_line + 2].strip().split()]
    # Read row pointers line
    row_ptr = [int(x) for x in lines[start_line + 3].strip().split()]

    assert len(vals) == nnz, f"Expected {nnz} values, got {len(vals)}"
    assert len(col_ind) == nnz
    assert len(row_ptr) == nbasis + 1, f"Expected {nbasis+1} row_ptr, got {len(row_ptr)}"

    return (rx, ry, rz, (vals, col_ind, row_ptr)), start_line + 4


def extract_submatrix(csr_data, row_start, row_end, col_start, col_end):
    """Extract a dense sub-matrix [row_start:row_end, col_start:col_end] from CSR data.
    Returns a dense numpy array of shape (row_end - row_start, col_end - col_start).
    """
    vals, col_ind, row_ptr = csr_data
    nrows = row_end - row_start
    ncols = col_end - col_start
    sub = np.zeros((nrows, ncols))

    for r_local, r_global in enumerate(range(row_start, row_end)):
        for idx in range(row_ptr[r_global], row_ptr[r_global + 1]):
            c_global = col_ind[idx]
            if col_start <= c_global < col_end:
                sub[r_local, c_global - col_start] = vals[idx]
    return sub


def read_csr_file(filepath):
    """Read a full CSR sparse matrix file.
    Returns list of (rx, ry, rz, csr_data) tuples.
    """
    with open(filepath) as f:
        lines = f.readlines()

    nbasis, nR, start = parse_csr_header(lines)
    matrices = []

    line_idx = start
    while line_idx < len(lines):
        result = parse_csr_matrix(lines, line_idx, nbasis)
        if result is None:
            break
        (rx, ry, rz, csr_data), line_idx = result
        matrices.append((rx, ry, rz, csr_data))

    return nbasis, matrices


# ---- NAO parsing (from cal_mayer_bond_order.py) ----

from dataclasses import dataclass, field, fields
from typing import Dict, Any, List


@dataclass
class AbacusNAO:
    element: str
    energy_cutoff: float
    radius: float
    Lmax: int
    l_orbs: List[int]
    orbs: List[Dict]
    mesh: int
    dr: float
    _dynamic_fields: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __setattr__(self, name, value):
        fixed_fields = {f.name for f in fields(self)}
        if name in fixed_fields:
            super().__setattr__(name, value)
        else:
            self._dynamic_fields[name] = value

    def __getattr__(self, name):
        if name in self._dynamic_fields:
            return self._dynamic_fields[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


def read_nao_file(nao_file):
    with open(nao_file) as f:
        content = f.readlines()

    summary_end_line = 0
    for i, line in enumerate(content):
        if 'SUMMARY  END' in line:
            summary_end_line = i
            break

    summary = content[1:summary_end_line - 1]
    element = summary[0].split()[-1]
    energy_cutoff = float(summary[1].split()[-1])
    radius = float(summary[2].split()[-1])
    Lmax = int(summary[3].split()[-1])
    n_l_orbs = [int(line.split()[-1]) for line in summary[4:]]

    mesh = int(content[summary_end_line + 2].split()[-1])
    dr = float(content[summary_end_line + 3].split()[-1])

    orb_data_start_lines = []
    for i, line in enumerate(content[summary_end_line + 4:]):
        if "Type" in line and "L" in line and "N" in line:
            orb_data_start_lines.append(i + summary_end_line + 4)

    orbs = []
    for orb_idx, start_line in enumerate(orb_data_start_lines):
        _, l, n = content[start_line + 1].split()
        l, n = int(l), int(n)
        if orb_idx == len(orb_data_start_lines) - 1:
            orb_data_raw = content[start_line + 2:]
        else:
            orb_data_raw = content[start_line + 2:orb_data_start_lines[orb_idx + 1]]
        orb_data = []
        for line in orb_data_raw:
            for num in line.split():
                orb_data.append(float(num))
        orbs.append({"l": l, "n": n, "orb_data": np.array(orb_data)})

    return AbacusNAO(element, energy_cutoff, radius, Lmax, n_l_orbs, orbs, mesh, dr)


def get_nao_basis_num(nao):
    nbas = 0
    for i, norb in enumerate(nao.l_orbs):
        nbas += (2 * i + 1) * norb
    return nbas


# ---- Pair selection (from cal_mayer_bond_order.py) ----

def calculate_minimum_distance(frac1, frac2, cell, max_range=3):
    import itertools
    diff = np.array(frac2) - np.array(frac1)
    diff -= np.round(diff)
    diff_cart = diff @ np.array(cell)
    min_dist_sq = float("inf")
    for n1, n2, n3 in itertools.product(range(-max_range, max_range + 1), repeat=3):
        translation = np.array([n1, n2, n3]) @ np.array(cell)
        dist_sq = np.sum((diff_cart + translation) ** 2)
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
    return np.sqrt(min_dist_sq)


def select_atom_pairs(stru, cutoff=None, pairs_str=None):
    if pairs_str is not None:
        pairs = []
        for token in pairs_str.split(","):
            token = token.strip()
            if not token:
                continue
            parts = token.split("-")
            if len(parts) != 2:
                raise ValueError(f"Invalid pair format: '{token}'. Use 'i-j'.")
            i, j = int(parts[0]) - 1, int(parts[1]) - 1
            if not (0 <= i < stru.natoms and 0 <= j < stru.natoms):
                raise ValueError(
                    f"Atom indices out of range: {i + 1}-{j + 1} (natoms={stru.natoms})")
            pairs.append((min(i, j), max(i, j)))
        return pairs

    if cutoff is not None:
        pairs = []
        for i in range(stru.natoms):
            for j in range(i + 1, stru.natoms):
                dist = calculate_minimum_distance(
                    stru.coords_direct[i], stru.coords_direct[j], stru.cell)
                if dist <= cutoff:
                    pairs.append((i, j))
        return pairs

    return None


# ---- Mayer bond order in R-space ----

def cal_mayer_bond_order_r_pair(iorb_atom1, iorb_atom2, dm_csr, ovlp_csr):
    """Compute Mayer bond order between two atoms from R-space CSR matrices.
    Extracts sub-blocks, multiplies, and sums the trace.
    """
    r1_start, r1_end = iorb_atom1[0], iorb_atom1[-1] + 1
    r2_start, r2_end = iorb_atom2[0], iorb_atom2[-1] + 1

    # Extract D_AB sub-block
    D_AB = extract_submatrix(dm_csr, r1_start, r1_end, r2_start, r2_end)
    # Extract S_AB sub-block
    S_AB = extract_submatrix(ovlp_csr, r1_start, r1_end, r2_start, r2_end)

    # PS = D_AB @ S_AB  (D is symmetric/Hermitian, but we use the extracted block)
    PS = D_AB @ S_AB

    # M = Σ_{μ,ν} PS[μ,ν] * PS[ν,μ]  (trace of PS^T * PS for this rectangular block)
    mayer_bond_order = np.sum(PS * PS.T)
    return mayer_bond_order


def cal_mayer_bond_order_r(abacusjob_dir, cutoff=None, pairs_str=None):
    """Calculate Mayer bond order from real-space density and overlap matrices."""
    from pathlib import Path
    from abacustest.lib_prepare.abacus import ReadInput
    from abacustest.lib_prepare.stru import AbacusSTRU

    input_params = ReadInput(os.path.join(Path(abacusjob_dir).absolute(), "INPUT"))
    nspin = input_params.get("nspin", 1)
    suffix = input_params.get("suffix", "ABACUS")
    out_dir = f"{abacusjob_dir}/OUT.{suffix}"

    # Parse structure and NAO
    stru_file = os.path.join(Path(abacusjob_dir).absolute(),
                             input_params.get("stru_file", "STRU"))
    stru = AbacusSTRU.read(stru_file)

    naos = {}
    orb_dir = input_params.get("orbital_dir") or str(Path(abacusjob_dir).absolute())
    for atom in stru.atoms:
        atom_nao = atom.orb
        if atom_nao not in naos:
            nao_file = os.path.join(orb_dir, atom_nao)
            naos[atom_nao] = read_nao_file(nao_file)

    atom_basis_nums = [get_nao_basis_num(naos[atom.orb]) for atom in stru.atoms]
    atom_orb_ranges = []
    start = 0
    for n in atom_basis_nums:
        end = start + n
        atom_orb_ranges.append((start, end))
        start = end

    # Select atom pairs
    selected_pairs = select_atom_pairs(stru, cutoff, pairs_str)
    if selected_pairs is not None:
        print(f"Selected {len(selected_pairs)} atom pair(s) for computation.")
    else:
        selected_pairs = [(i, j) for i in range(stru.natoms)
                          for j in range(i + 1, stru.natoms)]

    # Read DMR files
    dmr_file = os.path.join(out_dir, "data-DMR-sparse_SPIN1.csr")
    if not os.path.exists(dmr_file):
        raise FileNotFoundError(f"DMR file not found: {dmr_file}\n"
                                "Set 'out_dm1 1' in INPUT to generate it.")

    _, dmr_matrices = read_csr_file(dmr_file)
    print(f"Loaded DMR: {len(dmr_matrices)} R-vectors.")

    if nspin == 2:
        dmr_dn_file = os.path.join(out_dir, "data-DMR-sparse_SPIN2.csr")
        if not os.path.exists(dmr_dn_file):
            raise FileNotFoundError(f"DMR down-spin file not found: {dmr_dn_file}")
        _, dmr_dn_matrices = read_csr_file(dmr_dn_file)
    else:
        dmr_dn_matrices = None

    # Read SR files
    sr_file = os.path.join(out_dir, "data-SR-sparse_SPIN0.csr")
    if not os.path.exists(sr_file):
        raise FileNotFoundError(f"SR file not found: {sr_file}\n"
                                "Set 'out_mat_hs2 1' in INPUT to generate it.")

    _, sr_matrices = read_csr_file(sr_file)
    print(f"Loaded SR: {len(sr_matrices)} R-vectors.")

    # Compute per atom pair
    total_bond_order = {}
    for i, j in selected_pairs:
        total_bond_order[(i, j)] = 0.0

    nR = len(dmr_matrices)
    print(f"Computing Mayer bond order over {nR} R-vectors...")

    for iR, ((rx, ry, rz, dm_csr), (_, _, _, sr_csr)) in enumerate(
        zip(dmr_matrices, sr_matrices)
    ):
        for i, j in selected_pairs:
            s1, e1 = atom_orb_ranges[i]
            s2, e2 = atom_orb_ranges[j]
            bo = cal_mayer_bond_order_r_pair(
                list(range(s1, e1)), list(range(s2, e2)), dm_csr, sr_csr)
            total_bond_order[(i, j)] += bo

            if nspin == 2 and dmr_dn_matrices is not None:
                _, _, _, dm_dn_csr = dmr_dn_matrices[iR]
                bo_dn = cal_mayer_bond_order_r_pair(
                    list(range(s1, e1)), list(range(s2, e2)), dm_dn_csr, sr_csr)
                total_bond_order[(i, j)] += bo_dn

    if nspin == 2:
        for pair in total_bond_order:
            total_bond_order[pair] *= 2.0

    print("Mayer Bond Order (R-space):")
    for i, j in selected_pairs:
        bo = total_bond_order[(i, j)]
        if bo > 0.2:
            print(f"  {stru.atoms[i].label}{i + 1} - {stru.atoms[j].label}{j + 1}: {bo:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--abacusjob_dir", type=str, default="./",
                        help="ABACUS job directory")
    parser.add_argument("-c", "--cutoff", type=float, default=None,
                        help="Only compute pairs within cutoff distance (Angstrom)")
    parser.add_argument("-p", "--pairs", type=str, default=None,
                        help="Atom index pairs (1-indexed), e.g. '1-2,1-3'")
    args = parser.parse_args()

    if args.cutoff is not None and args.pairs is not None:
        parser.error("--cutoff and --pairs are mutually exclusive")

    print(f"Calculate Mayer bond order (R-space) for {args.abacusjob_dir}")
    cal_mayer_bond_order_r(args.abacusjob_dir,
                           cutoff=args.cutoff, pairs_str=args.pairs)
