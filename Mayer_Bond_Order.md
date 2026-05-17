# ABACUS Mayer 键级计算程序文档

## 概述

`cal_mayer_bond_order.py` 从 ABACUS LCAO 计算输出中计算 Mayer 键级。支持 gamma-only 和多 k 点两种模式，以及 nspin=1 和 nspin=2。

```
M_AB = Σ_{k∈IBZ} Σ_{μ∈A, ν∈B} (P_k S_k)_{μν} · (P_k S_k)_{νμ} / wk_k
```

## 理论公式

### Gamma-only（nspin=1）

```
M_AB = Σ_{μ∈A, ν∈B} (PS)_{μν} · (PS)_{νμ}
```
其中 P = 密度矩阵，S = 重叠矩阵，PS = P @ S。

### Gamma-only（nspin=2）

```
M_AB = 2 × Σ_{μ∈A, ν∈B} [(P↑S)_{μν}(P↑S)_{νμ} + (P↓S)_{μν}(P↓S)_{νμ}]
```
ABACUS 输出的 SPIN1_DM/SPIN2_DM 是自旋密度，乘以 2 恢复总密度。

### 多 k 点（nspin=1）

```
P_k = C_k · diag(√wg_k) · (C_k · diag(√wg_k))^H
M_AB^k = Σ_{μ∈A, ν∈B} (P_k S_k)_{μν} · (P_k S_k)_{νμ}
M_AB = Σ_{k∈IBZ} M_AB^k / wk_k
```
其中 wg_k 是 WFC_NAO_K 文件中的占据权重（已包含 k 点权重），wk_k 是 IBZ k 点权重。

### 多 k 点（nspin=2）

```
M_AB = Σ_{k∈IBZ} 2 × (M_AB^k↑ + M_AB^k↓) / wk_k
```
WFC 文件按 k1↑...kn↑, k1↓...kn↓ 排列。

## 程序结构

### 核心函数

### 文件结构（487 行）

| 行范围 | 内容 |
|---|---|
| 1-36 | 导入 + `AbacusNAO` 数据类 |
| 37-82 | `read_nao_file` — NAO 轨道解析 |
| 83-89 | `get_nao_basis_num` — 基函数计数 |
| 90-151 | `read_overlap_matrix` — 重叠矩阵（实数/复数） |
| 152-178 | `read_density_matrix` — SPIN*_DM 解析 |
| 179-195 | `cal_mayer_bond_order_between_atom_pair` — gamma-only 键级 |
| 196-233 | `read_wfc_nao_k` — 波函数解析 |
| 234-238 | `calculate_density_matrix_k` — P_k 构造 |
| 239-249 | `cal_mayer_bond_order_between_atom_pair_k` — 单 k 点键级 |
| 250-275 | `read_kpoint_weights` — IBZ 权重解析 |
| 276-294 | `calculate_minimum_distance` — 最小成像距离 |
| 295-338 | `select_atom_pairs` — 原子对筛选 |
| 339-476 | `cal_mayer_bond_order` — 主控逻辑 |
| 477-498 | `__main__` — 命令行接口 |

### 主控流程

```
cal_mayer_bond_order(abacusjob_dir)
  ├─ 读取 INPUT → nspin, gamma_only, suffix
  ├─ 读取 STRU → 原子信息、轨道文件
  ├─ 读取 NAO 文件 → 基函数计数、原子轨道范围映射
  │
  ├─ gamma_only == 1:
  │   ├─ 读取 SPIN1_DM（及 SPIN2_DM）
  │   └─ M = Σ(PS PS)
  │
  └─ 其他（multi-k）:
      ├─ glob WFC_NAO_K*.txt
      ├─ 读取 kpoints → IBZ 权重 wk
      │
      ├─ nspin=1:
      │   ├─ 逐 k: 读 WFC → 构建 P_k → 读 S_k
      │   └─ total += M_k / wk[k]
      │
      └─ nspin=2:
          ├─ WFC[0:nk] = 自旋上, WFC[nk:2nk] = 自旋下
          ├─ 逐 k: 分别构建 P↑_k, P↓_k
          └─ total += 2×(M↑_k + M↓_k) / wk[k]
```

## ABACUS 输出文件格式

### data-*-S（重叠矩阵）

```
<ndim> <row0_upper_tri> ...
<row1_upper_tri> ...
...
```
- 首值为矩阵维数 ndim
- 后续为上三角元素，按行排列
- gamma-only: 实数；multi-k: 复数 `(real,imag)` 格式

### SPIN*_DM（密度矩阵，仅 gamma-only）

```
<latName>
<lat0>
<cell vectors>
<atom labels>
<atom counts>
Direct
<fractional coords>

<nspin>
<fermi energy>
<nlocal> <nlocal>

<matrix values, 8 per row, row-major>
```

ABACUS 源码：`source/module_io/io_dmk.cpp`。`dmk_gen_fname` 对非 gamma-only 调用 `WARNING_QUIT`，因此多 k 点时不存在 SPIN*_DM 文件。

### WFC_NAO_K*.txt（波函数，多 k 点）

```
<kpoint index>
<k_x> <k_y> <k_z>
<nbands> (number of bands)
<nlocal> (number of orbitals)
<band 1> (band)
<eigenvalue> (Ry)
<occupation_weight> (Occupations)    ← 已包含 k 点权重
<coeff_real> <coeff_imag> ...
...
```

- nspin=2 时，文件按自旋分离：前 nk 个为自旋上，后 nk 个为自旋下
- wg 已包含 wk[ik]（IBZ k 点权重）

### kpoints（IBZ k 点及约化表）

```
nkstot now = <nk_ibz>
K-POINTS DIRECT COORDINATES
 KPOINTS    DIRECT_X    DIRECT_Y    DIRECT_Z  WEIGHT
       1    ...         ...        ...        <wk1>
...

nkstot = <nk_full>
K-POINTS REDUCTION ACCORDING TO SYMMETRY
 KPT    DIRECT_X    DIRECT_Y    DIRECT_Z   IBZ    DIRECT_X    DIRECT_Y    DIRECT_Z
   1    ...         ...        ...          1    ...         ...        ...
...
```

第一部分：IBZ k 点及权重；第二部分：原始 k 点 → IBZ 映射。

## 已验证的测试案例

所有 18 个组合（6 案例 × 3 筛选模式）均验证通过：

| 案例 | nspin | sym | k 点 | 默认 | `--cutoff` | `--pairs` | 键级 |
|---|---|---|---|---|---|---|---|
| H₂ | 1 | — | — | ✅ | ✅ | ✅ | 1.000139 |
| O₂ | 2 | — | — | ✅ | ✅ | ✅ | 2.009262 |
| Diamond | 1 | -1 | 64 | ✅ | ✅ | ✅ | 0.995160 |
| Diamond | 1 | 0 | 36 | ✅ | ✅ | ✅ | 0.995160 |
| CrI₃ | 2 | -1 | 9 | ✅ | ✅ | ✅ | 0.687904 |
| CrI₃ | 2 | 0 | 5 | ✅ | ✅ | ✅ | 0.687904 |

symmetry=-1 和 symmetry=0 结果完全一致，验证了 TRS 权重处理（Σ M_k / wk）的正确性。

## 对称性支持矩阵

### 工作原理

| 对称性 | k 点约化 | 波函数变换 | 原子映射 | Mayer 键级 |
|---|---|---|---|---|
| **sym=-1** | 无约化 | 无 | 不变 | ✅ 直接可用 |
| **sym=0** (TRS) | k→-k 约化 | ψ_{-k}=ψ*_k | 不变 | ✅ 权重加倍正确 |
| **sym=1** (全晶体) | k→Rk 约化 | ψ_{Rk}=D(R)·ψ_k | i→Si | ❌ 原子对映射错误 |

### 为什么 sym=1 不支持

Mayer 键级需要对特定原子对 (A,B) 做投影。全晶体对称下：

```
k' = R·k  (晶体旋转)
C_{Rk} = M(R,k)^† · C_k    ← M(R,k) 是 AO 旋转矩阵
```

对称伴侣 Rk 的密度矩阵应分配给映射后的原子对 (SA, SB)，而非 (A, B)。仅用 IBZ 权重乘法将所有对称伴侣的贡献错误地分配给了 IBZ 原子对，导致各向异性和数值偏差。

### ABACUS 内部的解决方案

`source/module_ri/module_exx_symmetry/symmetry_rotation.cpp` 中的 `restore_dm` 函数可为 EXX 计算展开全 BZ 密度矩阵：

```
D(Rk) = M(R,k)^† · D(k_IBZ) · M(R,k)
```

核心组件（约 800 行 C++）：
- **Wigner D 矩阵**：球谐函数旋转
- **Euler 角**：旋转矩阵分解
- **AO 旋转矩阵 M(R,k)**：原子轨道基组变换
- **原子索引映射**：对称操作下的原子等价关系

这些组件深嵌于 ABACUS 内部框架，无法在 Python 层复现。

### 推荐

- Mayer 键级：使用 **symmetry=0**（TRS only，精度等价于 -1）
- 最高精度：使用 **symmetry=-1**
- 不支持：symmetry=1（全晶体对称）

## 已修复的 Bug

| Bug | 原因 | 修复 |
|---|---|---|---|
| `out_mat_hs == 1` 断言失败 | `out_mat_hs 1 8` 被 ReadInput 解析为 `[1,8]` | 取首元素：`out_mat_hs[0] if isinstance` |
| `orbital_dir` 为 None 导致崩溃 | 参数存在但值为空时 `get('key', default)` 返回 None | `get('orbital_dir') or abacusjob_dir` |
| 多 k 点复数 S 矩阵解析崩溃 | `read_overlap_matrix` 仅处理实数 | 检测并解析 `(real,imag)` 格式 |
| 复数 PS 矩阵导致累加异常 | 复数乘法产生虚部 | 键级累加取 `.real` |
| `×nk` 修正因子对非均匀权重错误 | P_k 含 wk²，需除以 wk | 改为 `Σ M_k / wk` |
| 字典序 WFC 文件索引错误 | `sorted(glob)` 产生 K1,K10,... 而非 K1,K2,... | 用 `f"WFC_NAO_K{ik+1}.txt"` 直接构造路径 |

## 原子对筛选

通过 `--cutoff` 或 `--pairs` 参数可以在计算前筛选原子对，避免计算所有 n×(n-1)/2 对。

| 模式 | 用法 | 说明 |
|---|---|---|
| 默认（全部） | 不指定参数 | 计算所有 i<j 原子对 |
| 距离截断 | `--cutoff 2.0` | 仅计算最小成像距离 ≤ 2.0 Å 的原子对 |
| 指定原子对 | `--pairs "1-2,3-5"` | 仅计算指定的原子对（1-indexed） |

两种参数互斥。筛选不影响矩阵乘法（O(N³)），但减少原子对遍历（O(N²)）和避免输出噪声。

## k 空间 vs R 空间 Mayer 键级

当前实现使用 k 空间密度矩阵，计算的是参考原胞内原子对 (A,B) 的键级，但**所有周期性镜像的贡献通过 k 积分被隐式叠加**。对于 2 原子 diamond 原胞，4 条等价 C-C 键的贡献全部映射到唯一的 (C1,C2) 对上，M ≈ 4。

如需分离各周期性镜像的键级，需使用 ABACUS 的 R 空间矩阵（`data-DMR-sparse_*.csr`，`data-SR-sparse_*.csr`），当前版本不支持。

## 依赖

- Python 3
- numpy
- abacustest（`ReadInput`, `AbacusSTRU`）

## 使用方法

```bash
# 默认：计算所有原子对
python cal_mayer_bond_order.py -j /path/to/abacus_job_dir

# 距离截断：仅计算 2.0 Å 内的原子对
python cal_mayer_bond_order.py -j /path/to/abacus_job_dir --cutoff 2.0

# 指定原子对（1-indexed）
python cal_mayer_bond_order.py -j /path/to/abacus_job_dir --pairs "1-2,1-4"
```

前置条件：
- INPUT 中 `out_mat_hs 1`（输出重叠矩阵）
- gamma-only 模式需 `out_dm 1`（输出密度矩阵）
- 多 k 点模式需 `out_wfc_lcao 1`（输出波函数）
- NAO 轨道文件（.orb）在作业目录中

## 待补充的测试案例（nspin=2 + multi-k）

当前测试仅覆盖 CrI₃ 一种成键类型（Cr³⁺-I⁻ 离子键，BO≈0.69）。以下体系适合作为 nspin=2 + multi-k 下共价键级整数参考的验证案例：

### 简单分子（快速验证）

| 体系 | 基态 | 键型 | 预期 BO | 特点 |
|---|---|---|---|---|
| N₂ 三重态 | 强制 nspin=2 | N≡N | ≈3.0 | 最清晰的整数三重键参考 |
| O₂ multi-k | 自然 nspin=2 | O=O | ≈2.0 | 已有 gamma-only 结果对照 |
| S₂ 三重态 | 自然 nspin=2 | S=S | ≈2.0 | 重元素，验证基组收敛性 |
| NO 自由基 | 自然 nspin=2 | N=O | ≈2.5 | 奇电子分子，非整数但有标准值 |

### 真实材料（全面验证）

| 材料 | 磁性 | 结构 | 关键键 | 预期 BO | 选择理由 |
|---|---|---|---|---|---|
| VO₂ M1 | AFM singlet配对 | 单斜 P2₁/c | V–V dimer | ≈1.0 | 金属-绝缘体相变，V-V单键物理意义明确 |
| MnO | AFM (AF-II) | 岩盐 | Mn–O | ~0.3–0.5 | 经典 AFM 绝缘体，pd杂化共价 |
| NiO | AFM (AF-II) | 岩盐 | Ni–O | ~0.5–0.7 | 比 MnO 共价性更强，Mott绝缘体 |
| CrO₂ | FM 半金属 | 金红石 | Cr–O | ~0.8–1.0 | 强共价，自旋极化率100% |
| CoO | AFM (AF-II) | 岩盐 | Co–O | ~0.4–0.6 | 含自旋-轨道耦合 |
| Fe₃O₄ | 亚铁磁 | 反尖晶石 | Fe²⁺–O, Fe³⁺–O | 可区分 | 混合价态，两种Fe位 |

### 推荐优先级

1. **VO₂ M1** — V-V 二聚体键级 ≈1.0，金属-绝缘体相变的关键序参量
2. **MnO AFM** — AFM 绝缘体的标准模型，文献丰富
3. **CrO₂** — 半金属铁磁体，键级可作为自旋极化探针

## 文件版本

- 当前分支：`mayer_bo_multik`
- 文件名：`cal_mayer_bond_order.py`
- 行数：487 行
