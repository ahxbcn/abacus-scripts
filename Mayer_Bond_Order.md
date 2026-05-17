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

| 函数 | 职责 |
|---|---|
| `read_nao_file()` | 解析 NAO 轨道文件（.orb），获取基函数数量和轨道数据 |
| `get_nao_basis_num()` | 计算原子基函数数：Σ(2l+1)×norb |
| `read_overlap_matrix()` | 解析 data-*-S 文件（支持实数和复数格式） |
| `read_density_matrix()` | 解析 SPIN*_DM 文件（gamma-only） |
| `read_wfc_nao_k()` | 解析 WFC_NAO_K*.txt 波函数文件（多 k 点） |
| `calculate_density_matrix_k()` | 从波函数构建 P_k = C·diag(wg)·C^H |
| `read_kpoint_weights()` | 解析 kpoints 文件获取 IBZ 权重 |
| `cal_mayer_bond_order_between_atom_pair()` | gamma-only 键级计算（含 nspin=2） |
| `cal_mayer_bond_order_between_atom_pair_k()` | 单 k 点键级计算 |

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

| 案例 | 模式 | nspin | k 点数 | 键级 | 参考值 |
|---|---|---|---|---|---|
| H₂ | gamma-only | 1 | - | H-H: 1.000139 | 1.0 (单键) |
| O₂ | gamma-only | 2 | - | O-O: 2.009262 | 2.0 (双键) |
| Diamond sym=-1 | multi-k | 1 | 64 | C-C: 0.995160 | 1.0 (单键) |
| Diamond sym=0 | multi-k TRS | 1 | 36 | C-C: 0.995160 | 1.0 (单键) |
| CrI₃ sym=-1 | multi-k | 2 | 9 | Cr-I: 0.687904 | - |
| CrI₃ sym=0 | multi-k TRS | 2 | 5 | Cr-I: 0.687904 | - |

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
|---|---|---|
| `out_mat_hs == 1` 断言失败 | `out_mat_hs 1 8` 被 ReadInput 解析为 `[1,8]` | 取首元素：`out_mat_hs[0] if isinstance` |
| `orbital_dir` 为 None 导致崩溃 | 参数存在但值为空时 `get('key', default)` 返回 None | `get('orbital_dir') or abacusjob_dir` |
| 多 k 点复数 S 矩阵解析崩溃 | `read_overlap_matrix` 仅处理实数 | 检测并解析 `(real,imag)` 格式 |
| 复数 PS 矩阵导致累加异常 | 复数乘法产生虚部 | 键级累加取 `.real` |
| `×nk` 修正因子对非均匀权重错误 | P_k 含 wk²，需除以 wk | 改为 `Σ M_k / wk` |
| 字典序 WFC 文件索引错误 | `sorted(glob)` 产生 K1,K10,... 而非 K1,K2,... | 用 `f"WFC_NAO_K{ik+1}.txt"` 直接构造路径 |

## 依赖

- Python 3
- numpy
- abacustest（`ReadInput`, `AbacusSTRU`）

## 使用方法

```bash
python cal_mayer_bond_order.py -j /path/to/abacus_job_dir
```

前置条件：
- INPUT 中 `out_mat_hs 1`（输出重叠矩阵）
- gamma-only 模式需 `out_dm 1`（输出密度矩阵）
- 多 k 点模式需 `out_wfc_lcao 1`（输出波函数）
- NAO 轨道文件（.orb）在作业目录中

## 文件版本

- 当前分支：`mayer_bo_multik`
- 文件名：`cal_mayer_bond_order.py`
- 行数：417 行
