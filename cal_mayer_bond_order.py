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

if __name__ == '__main__':
    abacusjob_dir = "./"
    print(f"Calculate Mayer bond order for {abacusjob_dir}")
    # TODO: finish implementation

    H2_abacusjob_dir = "/mnt/e/profsoftfiles/abacusfiles/sp/H2_out_mat_hs"
    H2_ovlp_mat_file = f"{H2_abacusjob_dir}/OUT.ABACUS/data-0-S"
    H2_ovlp_mat = read_overlap_matrix(H2_ovlp_mat_file)
    print("H2 overlap matrix:", H2_ovlp_mat)
