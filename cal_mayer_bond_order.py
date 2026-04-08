"""
Calculate Mayer bond order for gamma-only calculations of ABACUS using LCAO basis.
"""
import numpy as np
from abacustest.lib_prepare.abacus import ReadInput

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

    print(empty_lines_idx)

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

if __name__ == '__main__':
    abacusjob_dir = "./"
    print(f"Calculate Mayer bond order for {abacusjob_dir}")
    # TODO: finish implementation

    H2_abacusjob_dir = "/mnt/e/profsoftfiles/abacusfiles/sp/H2_out_mat_hs"
    H2_ovlp_mat_file = f"{H2_abacusjob_dir}/OUT.ABACUS/data-0-S"
    H2_ovlp_mat = read_overlap_matrix(H2_ovlp_mat_file)
    print("H2 overlap matrix:", H2_ovlp_mat)

    dm_file = f"{H2_abacusjob_dir}/OUT.ABACUS/SPIN1_DM"
    dm = read_density_matrix(dm_file)
    print("H2 density matrix:", dm)
