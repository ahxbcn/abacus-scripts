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
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )


def read_nao_file(nao_file):
    with open(nao_file) as f:
        nao_file_content = f.readlines()

    summary_end_line = 0
    for linenum, line in enumerate(nao_file_content):
        if "SUMMARY  END" in line:
            summary_end_line = linenum
            break

    summary = nao_file_content[1 : summary_end_line - 1]
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
            orb_data_original = nao_file_content[
                orb_data_start_line + 2 : orb_data_start_lines[orb_idx + 1]
            ]

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
        nbas += (i * 2 + 1) * norb  # Use 5D 7F basis

    return nbas


def read_overlap_matrix(ovlp_mat_file):
    """
    Read overlap matrix from ovlp_mat_file.
    Supports both real and complex formats.
    """
    with open(ovlp_mat_file, "r") as f:
        lines = f.readlines()

    # Check if the file contains complex numbers (format like (real,imag))
    first_line = lines[0].strip()
    is_complex = "(" in first_line and "," in first_line and ")" in first_line

    all_nums = []
    for line in lines:
        if not line.strip():
            continue
        if is_complex:
            # Parse complex numbers in (real,imag) format
            line = line.replace("(", " ").replace(")", " ").replace(",", " ")
            nums = list(map(float, line.split()))
            all_nums.extend(nums)
        else:
            vals = list(map(float, line.split()))
            all_nums.extend(vals)

    if is_complex:
        ndim = int(all_nums[0])
        # First number is dimension, rest are pairs of (real, imag)
        complex_vals = []
        for i in range(1, len(all_nums), 2):
            complex_vals.append(complex(all_nums[i], all_nums[i + 1]))

        S = np.zeros((ndim, ndim), dtype=np.complex128)
        idx = 0
        for i in range(ndim):
            for j in range(i, ndim):
                S[i, j] = complex_vals[idx]
                S[j, i] = complex_vals[idx].conjugate()  # Complex Hermitian matrix
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
    for iline in range(empty_lines_idx[-1] + 1, len(lines)):
        line_vals = [float(x) for x in lines[iline].split()]
        vals.extend(line_vals)

    assert len(vals) == ndim * ndim
    dm_mat = np.array(vals).reshape(ndim, ndim)

    return dm_mat


def cal_mayer_bond_order_between_atom_pair(
    iorb_atom1, iorb_atom2, ovlp_mat, dm, dm_dn=None
):
    """
    Calculate Mayer bond order between two atoms.
    """
    mayer_bond_order = 0
    if dm_dn is None:  # nspin = 1 case
        PS = dm @ ovlp_mat
        for iorb in iorb_atom1:
            for jorb in iorb_atom2:
                mayer_bond_order += PS[iorb, jorb] * PS[jorb, iorb]
    else:  # nspin = 2 case
        PS = dm @ ovlp_mat
        PS_dn = dm_dn @ ovlp_mat
        for iorb in iorb_atom1:
            for jorb in iorb_atom2:
                mayer_bond_order += (
                    PS[iorb, jorb] * PS[jorb, iorb]
                    + PS_dn[iorb, jorb] * PS_dn[jorb, iorb]
                )
        mayer_bond_order *= 2

    return mayer_bond_order


def cal_mayer_bond_order_between_atom_pair_k(iorb_atom1, iorb_atom2, ovlp_mat, dm):
    """
    Calculate Mayer bond order contribution from a single k-point between two atoms.
    """
    mayer_bond_order_k = 0
    PS = dm @ ovlp_mat
    for iorb in iorb_atom1:
        for jorb in iorb_atom2:
            mayer_bond_order_k += PS[iorb, jorb] * PS[jorb, iorb]
    return mayer_bond_order_k


def calculate_minimum_distance_fractional(coords1, coords2, cell, max_range=10):
    """
    计算两个原子之间的最小距离，考虑周期性边界条件。

    Args:
        coords1: 第一个原子的分数坐标 [x, y, z]
        coords2: 第二个原子的分数坐标 [x, y, z]
        cell: 晶胞矩阵 [[ax, ay, az], [bx, by, bz], [cx, cy, cz]] (Angstrom)
        max_range: 搜索的最大晶胞范围

    Returns:
        最小距离 (Angstrom)
    """
    import itertools

    # 将分数坐标差转换到 [-0.5, 0.5] 范围内
    diff = np.array(coords2) - np.array(coords1)
    diff = diff - np.round(diff)

    # 转换到笛卡尔坐标
    cell_matrix = np.array(cell)
    diff_cart = diff @ cell_matrix

    # 搜索附近的晶胞，寻找最小距离
    min_dist_sq = float("inf")

    # 晶胞搜索范围通常只需要考虑相邻的晶胞
    for n1, n2, n3 in itertools.product(
        range(-max_range, max_range + 1),
        range(-max_range, max_range + 1),
        range(-max_range, max_range + 1),
    ):
        # 晶胞平移矢量
        translation = np.array([n1, n2, n3]) @ cell_matrix
        dist_sq = np.sum((diff_cart + translation) ** 2)
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq

    return np.sqrt(min_dist_sq)


def find_bonded_atom_pairs(stru, cutoff_distance=3.5, max_cell_range=5):
    """
    找出距离小于截断值的所有原子对。

    Args:
        stru: AbacusSTRU 对象
        cutoff_distance: 距离截断值 (Angstrom)
        max_cell_range: 晶胞搜索范围

    Returns:
        List of tuples (i, j) 表示需要计算键级的原子对索引
    """
    natoms = stru.natoms
    coords_direct = stru.coords_direct  # 分数坐标
    cell = stru.cell  # 晶胞矩阵

    bonded_pairs = []

    # 遍历所有原子对 (只计算 i < j 以避免重复)
    for i in range(natoms):
        for j in range(i + 1, natoms):
            dist = calculate_minimum_distance_fractional(
                coords_direct[i], coords_direct[j], cell, max_range=max_cell_range
            )
            if dist < cutoff_distance:
                bonded_pairs.append((i, j))

    return bonded_pairs


def cal_mayer_bond_order(abacusjob_dir, cutoff_distance=3.5):
    """
    Calculate Mayer bond order from ABACUS calculation output.

    Args:
        abacusjob_dir: ABACUS 计算任务目录
        cutoff_distance: 距离截断值 (Angstrom)，只计算距离小于此值的原子对之间的键级
    """
    from pprint import pprint
    from pathlib import Path
    import glob
    from abacustest.lib_prepare.abacus import ReadInput
    from abacustest.lib_prepare.stru import AbacusSTRU

    input_params = ReadInput(os.path.join(Path(abacusjob_dir).absolute(), "INPUT"))
    nspin = input_params.get("nspin", 1)
    gamma_only = input_params.get("gamma_only", 1)

    suffix = input_params.get("suffix", "ABACUS")
    out_dir = f"{abacusjob_dir}/OUT.{suffix}"

    stru_file = os.path.join(
        Path(abacusjob_dir).absolute(), input_params.get("stru_file", "STRU")
    )
    stru = AbacusSTRU.read(stru_file)
    # Read NAOs for each atoms
    naos = {}
    orb_dir = input_params.get("orbital_dir", "./")
    # If orbital_dir is relative, make it relative to abacusjob_dir
    if not os.path.isabs(orb_dir):
        orb_dir = os.path.join(abacusjob_dir, orb_dir)
    for atom in stru.atoms:
        atom_nao = atom.orb
        if atom_nao not in naos.keys():
            nao_file = os.path.join(orb_dir, atom_nao)
            nao = read_nao_file(nao_file)
            naos[atom_nao] = nao

    atom_basis_nums = []
    for atom in stru.atoms:
        atom_basis_nums.append(get_nao_basis_num(naos[atom.orb]))

    # Find bonded atom pairs based on distance cutoff
    print(f"\nFinding atom pairs within {cutoff_distance} Angstrom...")
    bonded_pairs = find_bonded_atom_pairs(stru, cutoff_distance=cutoff_distance)
    print(
        f"Found {len(bonded_pairs)} bonded pairs out of {stru.natoms * (stru.natoms - 1) // 2} total pairs"
    )

    # Check if we have WFC_NAO_K*.txt files to determine if it's multi-k-point
    import glob

    wfc_files = glob.glob(os.path.join(out_dir, "WFC_NAO_K*.txt"))
    is_multi_k = len(wfc_files) > 0

    if gamma_only == 1 and not is_multi_k:
        assert input_params.get("out_mat_hs", 1) == 1
        assert input_params.get("out_dm", 1) == 1
        ovlp_mat_file = f"{out_dir}/data-0-S"
        ovlp_mat = read_overlap_matrix(ovlp_mat_file)
        dm_file = f"{out_dir}/SPIN1_DM"
        dm = read_density_matrix(dm_file)
        if nspin == 2:
            dm_dn_file = f"{out_dir}/SPIN2_DM"
            dm_dn = read_density_matrix(dm_dn_file)
        else:
            dm_dn = None

        print("Mayer Bond Order (Gamma-only):")
        for i, j in bonded_pairs:
            iorb_atom1 = [
                iorb
                for iorb in range(
                    sum(atom_basis_nums[:i]), sum(atom_basis_nums[: i + 1])
                )
            ]
            iorb_atom2 = [
                iorb
                for iorb in range(
                    sum(atom_basis_nums[:j]), sum(atom_basis_nums[: j + 1])
                )
            ]

            mayer_bond_order = cal_mayer_bond_order_between_atom_pair(
                iorb_atom1, iorb_atom2, ovlp_mat, dm, dm_dn
            )
            atomtype1, atomtype2 = stru.atoms[i].label, stru.atoms[j].label
            if mayer_bond_order > 0.2:
                print(f"{atomtype1}{i + 1} - {atomtype2}{j + 1}: {mayer_bond_order}")
    else:
        # Multi-k-point calculation
        # Find all k-point files
        wfc_files = sorted(glob.glob(os.path.join(out_dir, "WFC_NAO_K*.txt")))

        if not wfc_files:
            raise FileNotFoundError(
                "No WFC_NAO_K*.txt files found for multi-k-point calculation"
            )

        # Get number of k-points
        nk = len(wfc_files)
        print(f"Found {nk} k-points")

        # Initialize total bond order dictionary
        total_bond_order = {pair: 0.0 for pair in bonded_pairs}

        for ik in range(nk):
            # Read wavefunction and k-point (note: file numbering starts from 1)
            wfc_file = os.path.join(out_dir, f"WFC_NAO_K{ik + 1}.txt")
            wfc, wg, _, _, _ = read_wfc_nao_k(wfc_file)

            # Read overlap matrix for this k-point (note: file numbering starts from 0)
            ovlp_file = os.path.join(out_dir, f"data-{ik}-S")
            ovlp_mat = read_overlap_matrix(ovlp_file)

            # Calculate density matrix for this k-point
            dm_k = calculate_density_matrix_k(wfc, wg)

            # Calculate contribution from this k-point for each bonded atom pair
            for i, j in bonded_pairs:
                print(f"Calculating contribution from k-point {ik} for atom pair {i} {j}")
                iorb_atom1 = [
                    iorb
                    for iorb in range(
                        sum(atom_basis_nums[:i]), sum(atom_basis_nums[: i + 1])
                    )
                ]
                iorb_atom2 = [
                    iorb
                    for iorb in range(
                        sum(atom_basis_nums[:j]), sum(atom_basis_nums[: j + 1])
                    )
                ]

                bond_order_k = cal_mayer_bond_order_between_atom_pair_k(
                    iorb_atom1, iorb_atom2, ovlp_mat, dm_k
                )
                total_bond_order[(i, j)] += bond_order_k

        # Multiply by nk as suggested by user's observation
        for key in total_bond_order:
            total_bond_order[key] *= nk

        # Print results
        print("Mayer Bond Order (Multi-k-point):")
        for (i, j), bo in total_bond_order.items():
            atomtype1, atomtype2 = stru.atoms[i].label, stru.atoms[j].label
            bo_real = bo.real if isinstance(bo, np.complex128) else bo
            if bo_real > 0.2:
                print(f"{atomtype1}{i + 1} - {atomtype2}{j + 1}: {bo_real:10f}")


def read_wfc_nao_k(file_path):
    """
    Read txt format wavefunction file WFC_NAO_K*.txt
    """
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    nbands = None
    nlocal = None
    kvec_c = None

    for line in lines:
        if line.endswith("(number of bands)"):
            nbands = int(line.split()[0])
        elif line.endswith("(number of orbitals)"):
            nlocal = int(line.split()[0])
        elif not line.endswith(")") and line.count(" ") == 2:
            kvec_c = np.array([float(x) for x in line.split()])

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
            continue
        elif line.endswith("(Ry)"):
            continue
        elif line.endswith("(Occupations)"):
            wg[ib] = float(line.split()[0])
            continue
        elif reading_band and not line.endswith(")"):
            # Read wavefunction coefficients
            if line.count(" ") >= 1:
                nums = line.split()
                # Complex stored in: real1 imag1 real2 imag2 ...
                c = [
                    complex(float(nums[i]), float(nums[i + 1]))
                    for i in range(0, len(nums), 2)
                ]
                for i in range(len(c)):
                    if ilocal < nlocal:
                        wfc[ilocal, ib] = c[i]
                        ilocal += 1

    return wfc, wg, kvec_c, nbands, nlocal


def calculate_density_matrix_k(wfc, wg):
    """
    Calculate density matrix from wave function in LCAO basis.
    """
    wfc_weighted = wfc * np.sqrt(wg)[np.newaxis, :]
    dm = wfc_weighted @ wfc_weighted.conj().T

    return dm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-j",
        "--abacusjob_dir",
        type=str,
        default="./",
        help="ABACUS job directory to calculate Mayer bond order",
    )
    parser.add_argument(
        "-c",
        "--cutoff_distance",
        type=float,
        default=3.5,
        help="Cutoff distance (Angstrom) for finding bonded atom pairs",
    )
    args = parser.parse_args()

    abacusjob_dir = args.abacusjob_dir
    print(f"Calculate Mayer bond order for {abacusjob_dir}")
    cal_mayer_bond_order(abacusjob_dir, cutoff_distance=args.cutoff_distance)
