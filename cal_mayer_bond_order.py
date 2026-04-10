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


def read_kpoints_file(kpoints_file):
    """
    Read ABACUS kpoints file to get all k-point coordinates and weights,
    including symmetry reduction information.

    Returns:
        dict with keys:
            'reduced_kpoints': list of reduced k-points
            'full_kpoints': list of full k-points
            'kpt_to_ibz': dict mapping full k-point index to IBZ k-point index
    """
    reduced_kpoints = []
    full_kpoints = []
    kpt_to_ibz = {}

    with open(kpoints_file, "r") as f:
        lines = f.readlines()

    # Read first section: K-POINTS DIRECT COORDINATES (reduced k-points)
    found_direct_header = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("K-POINTS DIRECT COORDINATES"):
            found_direct_header = True
            continue

        # Stop at empty line (end of direct coordinates section)
        if found_direct_header and stripped == "":
            break

        if not found_direct_header:
            continue

        # Parse k-point data lines
        parts = stripped.split()
        if len(parts) >= 5:  # KPOINTS DIRECT_X DIRECT_Y DIRECT_Z WEIGHT
            try:
                kpt_idx = int(parts[0])
                dx = float(parts[1])
                dy = float(parts[2])
                dz = float(parts[3])
                weight = float(parts[4])
                reduced_kpoints.append(
                    {"index": kpt_idx, "direct": (dx, dy, dz), "weight": weight}
                )
            except ValueError:
                continue

    # Read second section: K-POINTS REDUCTION ACCORDING TO SYMMETRY
    found_reduction_header = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("K-POINTS REDUCTION ACCORDING TO SYMMETRY"):
            found_reduction_header = True
            continue

        if not found_reduction_header:
            continue

        # Parse full k-point and IBZ mapping
        parts = stripped.split()
        if (
            len(parts) >= 7
        ):  # KPT DIRECT_X DIRECT_Y DIRECT_Z IBZ DIRECT_X DIRECT_Y DIRECT_Z
            try:
                kpt_idx = int(parts[0])
                dx = float(parts[1])
                dy = float(parts[2])
                dz = float(parts[3])
                ibz_idx = int(parts[4])

                full_kpoints.append({"index": kpt_idx, "direct": (dx, dy, dz)})
                kpt_to_ibz[kpt_idx] = ibz_idx
            except ValueError:
                continue

    return {
        "reduced_kpoints": reduced_kpoints,
        "full_kpoints": full_kpoints,
        "kpt_to_ibz": kpt_to_ibz,
    }


def find_negative_k(kvec, kpoints_list, tol=1e-6):
    """
    Find the index of the k-point (-kx, -ky, -kz) in kpoints_list.
    """
    target = (-kvec[0], -kvec[1], -kvec[2])
    for idx, kpt in enumerate(kpoints_list):
        d = kpt["direct"]
        # Check if coordinates match within tolerance, considering periodicity (mod 1.0)
        diff0 = abs((d[0] - target[0]) % 1.0)
        diff1 = abs((d[1] - target[1]) % 1.0)
        diff2 = abs((d[2] - target[2]) % 1.0)
        if (
            (diff0 < tol or (1.0 - diff0) < tol)
            and (diff1 < tol or (1.0 - diff1) < tol)
            and (diff2 < tol or (1.0 - diff2) < tol)
        ):
            return idx
    return -1


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

    out_wfc_lcao = input_params.get("out_wfc_lcao", 1)

    import glob

    if out_wfc_lcao == 2:
        wfc_files = sorted(glob.glob(os.path.join(out_dir, "WFC_NAO_K*.dat")))
        wfc_suffix = ".dat"
    else:
        wfc_files = sorted(glob.glob(os.path.join(out_dir, "WFC_NAO_K*.txt")))
        wfc_suffix = ".txt"

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
        wfc_suffix_stripped = wfc_suffix.replace(".", "")
        wfc_files = sorted(
            glob.glob(os.path.join(out_dir, f"WFC_NAO_K*.{wfc_suffix_stripped}"))
        )

        if not wfc_files:
            raise FileNotFoundError(
                f"No WFC_NAO_K*.{wfc_suffix_stripped} files found for multi-k-point calculation"
            )

        # Read kpoints file to get full list of k-points and symmetry info
        kpoints_file = os.path.join(out_dir, "kpoints")
        use_time_reversal = False
        kpoints_data = None
        full_kpoints = []
        kvec_to_data = {}
        unique_kvecs = []
        actual_nk = 0
        kvec_to_idx = {}

        if os.path.exists(kpoints_file):
            kpoints_data = read_kpoints_file(kpoints_file)
            reduced_kpoints = kpoints_data["reduced_kpoints"]
            full_kpoints = kpoints_data["full_kpoints"]
            kpt_to_ibz = kpoints_data["kpt_to_ibz"]

            print(
                f"Read {len(reduced_kpoints)} reduced k-points, {len(full_kpoints)} full k-points from kpoints file"
            )

            # Check if WFC files are fewer than full k-points (symmetry reduction)
            if len(wfc_files) < len(full_kpoints):
                use_time_reversal = True
                print(
                    f"Detected symmetry reduction: {len(wfc_files)} WFC files, {len(full_kpoints)} total k-points"
                )
                print("Using time reversal symmetry to fill in missing k-points")

        # Initialize total bond order dictionary
        total_bond_order = {pair: 0.0 for pair in bonded_pairs}

        if use_time_reversal:
            # Build a map from IBZ k-vector to (dm, ovlp)
            # For time-reversal symmetry, we need to map IBZ k-points to full k-points
            # First, read all available WFC and S files
            ibz_kvec_to_data = {}
            for wfc_file in wfc_files:
                # Read k-vector from WFC file
                if wfc_suffix == ".dat":
                    wfc, wg, kvec_np, _, _ = read_wfc_nao_k_dat(wfc_file)
                    kvec = tuple(kvec_np)  # Convert to tuple for hashing
                    file_basename = os.path.basename(wfc_file)
                    file_idx_str = file_basename.replace("WFC_NAO_K", "").replace(
                        ".dat", ""
                    )
                else:
                    with open(wfc_file, "r") as f:
                        lines = f.readlines()
                        kvec_str = lines[1].strip()
                        kvec = tuple(float(x) for x in kvec_str.split())

                    file_basename = os.path.basename(wfc_file)
                    file_idx_str = file_basename.replace("WFC_NAO_K", "").replace(
                        f".{wfc_suffix.replace('.', '')}", ""
                    )
                file_idx = int(file_idx_str) - 1  # 0-based

                dm_k = calculate_density_matrix_k(wfc, wg)

                ovlp_file = os.path.join(out_dir, f"data-{file_idx}-S")
                ovlp_mat = read_overlap_matrix(ovlp_file)

                ibz_kvec_to_data[kvec] = (dm_k, ovlp_mat)

            # Now process all full k-points
            for full_kpt in full_kpoints:
                full_kvec = full_kpt["direct"]
                full_kpt_idx = full_kpt["index"]

                # Find corresponding IBZ k-point index
                ibz_kpt_idx = kpt_to_ibz[full_kpt_idx]

                # Get IBZ k-point's coordinates
                ibz_kpt = reduced_kpoints[ibz_kpt_idx - 1]  # IBZ indices start at 1
                ibz_kvec = ibz_kpt["direct"]

                dm_k = None
                ovlp_mat = None

                # First check if we have this IBZ k-point directly
                if ibz_kvec in ibz_kvec_to_data:
                    dm_k, ovlp_mat = ibz_kvec_to_data[ibz_kvec]
                else:
                    # Check if we can get it via time reversal: -k point
                    neg_ibz_kvec = (-ibz_kvec[0], -ibz_kvec[1], -ibz_kvec[2])

                    # Find the IBZ k-point for -kvec
                    neg_ibz_kpt_idx = -1
                    for idx, kpt in enumerate(reduced_kpoints):
                        d = kpt["direct"]
                        # Check if coordinates match within tolerance, considering periodicity
                        tol = 1e-6
                        diff0 = abs((d[0] - neg_ibz_kvec[0]) % 1.0)
                        diff1 = abs((d[1] - neg_ibz_kvec[1]) % 1.0)
                        diff2 = abs((d[2] - neg_ibz_kvec[2]) % 1.0)
                        if (
                            (diff0 < tol or (1.0 - diff0) < tol)
                            and (diff1 < tol or (1.0 - diff1) < tol)
                            and (diff2 < tol or (1.0 - diff2) < tol)
                        ):
                            neg_ibz_kpt_idx = idx
                            break

                    if neg_ibz_kpt_idx != -1:
                        neg_ibz_kpt = reduced_kpoints[neg_ibz_kpt_idx]
                        neg_kvec = neg_ibz_kpt["direct"]
                        if neg_kvec in ibz_kvec_to_data:
                            # Apply time reversal: P(-k) = P(k)^*, S(-k) = S(k)^*
                            dm_neg, ovlp_neg = ibz_kvec_to_data[neg_kvec]
                            dm_k = dm_neg.conj()
                            ovlp_mat = ovlp_neg.conj()

                if dm_k is not None and ovlp_mat is not None:
                    # Calculate contribution from this k-point
                    for i, j in bonded_pairs:
                        iorb_atom1 = list(
                            range(
                                sum(atom_basis_nums[:i]), sum(atom_basis_nums[: i + 1])
                            )
                        )
                        iorb_atom2 = list(
                            range(
                                sum(atom_basis_nums[:j]), sum(atom_basis_nums[: j + 1])
                            )
                        )
                        bond_order_k = cal_mayer_bond_order_between_atom_pair_k(
                            iorb_atom1, iorb_atom2, ovlp_mat, dm_k
                        )
                        total_bond_order[(i, j)] += bond_order_k

            # Multiply by total k-point count TWICE as done in symmetry=-1 case
            nk_total = len(full_kpoints)
            for key in total_bond_order:
                total_bond_order[key] *= nk_total * nk_total

        elif nspin == 2 and len(wfc_files) == 2 * actual_nk:
            # Spin-polarized: first half are spin-up, second half are spin-down
            # For spin-up: file indices 1..actual_nk, S indices 0..actual_nk-1
            # For spin-down: file indices actual_nk+1..2*actual_nk, S indices actual_nk..2*actual_nk-1
            print("Detected nspin=2: processing spin-up and spin-down separately")

            # Process each unique k-point
            for kvec in unique_kvecs:
                files_for_k = kvec_to_idx[kvec]

                # Sort by file index to separate spin-up (lower index) from spin-down (higher index)
                files_for_k_sorted = sorted(files_for_k, key=lambda x: x[1])

                # First file is spin-up, second is spin-down
                spin_up_file = files_for_k_sorted[0][0]
                spin_up_idx = files_for_k_sorted[0][1]  # 1 to actual_nk

                spin_down_file = files_for_k_sorted[1][0]
                spin_down_idx = files_for_k_sorted[1][1]  # actual_nk+1 to 2*actual_nk

                # Process spin-up
                ovlp_file_up = os.path.join(out_dir, f"data-{spin_up_idx - 1}-S")
                ovlp_mat_up = read_overlap_matrix(ovlp_file_up)
                if wfc_suffix == ".dat":
                    wfc_up, wg_up, _, _, _ = read_wfc_nao_k_dat(spin_up_file)
                else:
                    wfc_up, wg_up, _, _, _ = read_wfc_nao_k(spin_up_file)
                dm_k_up = calculate_density_matrix_k(wfc_up, wg_up)

                # Process spin-down
                ovlp_file_dn = os.path.join(out_dir, f"data-{spin_down_idx - 1}-S")
                ovlp_mat_dn = read_overlap_matrix(ovlp_file_dn)
                if wfc_suffix == ".dat":
                    wfc_dn, wg_dn, _, _, _ = read_wfc_nao_k_dat(spin_down_file)
                else:
                    wfc_dn, wg_dn, _, _, _ = read_wfc_nao_k(spin_down_file)
                dm_k_dn = calculate_density_matrix_k(wfc_dn, wg_dn)

                for i, j in bonded_pairs:
                    iorb_atom1 = list(
                        range(sum(atom_basis_nums[:i]), sum(atom_basis_nums[: i + 1]))
                    )
                    iorb_atom2 = list(
                        range(sum(atom_basis_nums[:j]), sum(atom_basis_nums[: j + 1]))
                    )

                    # Calculate spin-up contribution
                    bond_order_up = cal_mayer_bond_order_between_atom_pair_k(
                        iorb_atom1, iorb_atom2, ovlp_mat_up, dm_k_up
                    )

                    # Calculate spin-down contribution
                    bond_order_dn = cal_mayer_bond_order_between_atom_pair_k(
                        iorb_atom1, iorb_atom2, ovlp_mat_dn, dm_k_dn
                    )

                    total_bond_order[(i, j)] += bond_order_up + bond_order_dn

            # Multiply by actual_nk * nspin as before
            for key in total_bond_order:
                total_bond_order[key] *= actual_nk * nspin

        else:
            # Non-spin-polarized or gamma-only multi-k
            # Get number of k-points
            nk = len(wfc_files)
            print(f"Found {nk} k-points")

            for ik in range(nk):
                # Read wavefunction and k-point (note: file numbering starts from 1)
                wfc_file = os.path.join(out_dir, f"WFC_NAO_K{ik + 1}{wfc_suffix}")
                if wfc_suffix == ".dat":
                    wfc, wg, _, _, _ = read_wfc_nao_k_dat(wfc_file)
                else:
                    wfc, wg, _, _, _ = read_wfc_nao_k(wfc_file)

                # Read overlap matrix for this k-point (note: file numbering starts from 0)
                ovlp_file = os.path.join(out_dir, f"data-{ik}-S")
                ovlp_mat = read_overlap_matrix(ovlp_file)

                # Calculate density matrix for this k-point
                dm_k = calculate_density_matrix_k(wfc, wg)

                # Calculate contribution from this k-point for each bonded atom pair
                for i, j in bonded_pairs:
                    print(
                        f"Calculating contribution from k-point {ik} for atom pair {i} {j}"
                    )
                    iorb_atom1 = list(
                        range(sum(atom_basis_nums[:i]), sum(atom_basis_nums[: i + 1]))
                    )
                    iorb_atom2 = list(
                        range(sum(atom_basis_nums[:j]), sum(atom_basis_nums[: j + 1]))
                    )

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


def read_wfc_nao_k_dat(file_path):
    """
    Read binary dat format wavefunction file WFC_NAO_K*.dat
    """
    import struct

    with open(file_path, "rb") as f:
        data = f.read()

    pos = 0

    def read_int():
        nonlocal pos
        val = struct.unpack("@i", data[pos : pos + 4])[0]
        pos += 4
        return val

    def read_double():
        nonlocal pos
        val = struct.unpack("@d", data[pos : pos + 8])[0]
        pos += 8
        return val

    ik_plus_1 = read_int()
    kvec_c = np.array([read_double(), read_double(), read_double()])
    nbands = read_int()
    nlocal = read_int()

    wfc = np.zeros((nlocal, nbands), dtype=np.complex128)
    wg = np.zeros(nbands)

    for i in range(nbands):
        band_idx = read_int()
        ekb = read_double()
        wg[band_idx - 1] = read_double()

        for j in range(nlocal):
            real_part = read_double()
            imag_part = read_double()
            wfc[j, band_idx - 1] = complex(real_part, imag_part)

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
