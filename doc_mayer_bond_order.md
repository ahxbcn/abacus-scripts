# Mayer键级计算代码说明

## 文件位置

`/home/ahxb/repos/abacus-scripts/cal_mayer_bond_order.py`

## 分支

当前代码位于 `mayer_bond_order` 分支。

## 功能概述

计算ABACUS LCAO基组计算中的Mayer键级。

## 计算模式

### Gamma-only (单k点) ✅

使用Gamma点（k=0）处的密度矩阵和重叠矩阵进行计算。

**INPUT设置**：
```
gamma_only = 1
out_mat_hs = 1
out_dm = 1
```

### 多k点采样 ✅

支持完整的多k点采样计算，包含以下特性：
- 非自旋极化 (nspin=1) 多k点
- 自旋极化 (nspin=2) 多k点
- 时间反演对称性（IBZ约化k点自动填充）
- 复数重叠矩阵读取
- 支持txt和dat两种波函数输出格式

**INPUT设置**：
```
gamma_only = 0  # 或不设置
out_mat_hs = 1
out_wfc_lcao = 1  # 输出txt格式波函数
# 或
out_wfc_lcao = 2  # 输出dat格式(二进制)波函数
```

## 核心功能

### 1. 数据读取

**Gamma-only模式**：
- 重叠矩阵：`OUT.{suffix}/data-0-S`
- 密度矩阵：`SPIN1_DM` / `SPIN2_DM`

**多k点模式**：
- 波函数：`WFC_NAO_K{ik}.txt` (txt格式，复数LCAO系数) 或 `WFC_NAO_K{ik}.dat` (二进制格式)
- 重叠矩阵：`data-{ik}-S` (复数格式)
- kpoints文件：获取k点权重和对称性信息

**通用**：
- NAO轨道文件：获取基组信息
- STRU文件：获取原子结构

### 2. Mayer键级计算

公式（多k点）：

$$BO = \sum_{k} \sum_{i \in A} \sum_{j \in B} (P_k S_k)_{ij} (P_k S_k)_{ji}$$

需乘以k点数量获得正确结果。

### 3. 优化功能

- **距离截断**：通过 `cutoff_distance` 参数只计算近邻原子对
- **时间反演对称性**：利用-k点自动推算缺失的k点

### 4. 结果输出

- 格式：`元素类型1-元素类型2: 键级值`

## 命令行用法

```bash
# 基本用法
python cal_mayer_bond_order.py -j /path/to/abacus/job

# 指定截断距离
python cal_mayer_bond_order.py -j /path/to/abacus/job -c 3.5
```

## 函数列表

| 函数 | 行号 | 说明 |
|------|------|------|
| `read_nao_file` | 40 | 读取NAO轨道文件 |
| `get_nao_basis_num` | 91 | 获取基组函数数量 |
| `read_overlap_matrix` | 102 | 读取重叠矩阵(支持复数) |
| `read_density_matrix` | 156 | 读取密度矩阵 |
| `cal_mayer_bond_order_between_atom_pair` | 184 | Gamma-only键级计算 |
| `cal_mayer_bond_order_between_atom_pair_k` | 210 | 单k点键级计算 |
| `read_kpoints_file` | 293 | 读取k点信息 |
| `read_wfc_nao_k` | ~746 | 读取txt格式波函数文件 |
| `read_wfc_nao_k_dat` | ~783 | 读取dat格式(二进制)波函数文件 |
| `calculate_density_matrix_k` | ~823 | 从波函数计算密度矩阵 |
| `find_bonded_atom_pairs` | 263 | 基于距离找键合原子对 |
| `cal_mayer_bond_order` | 396 | 主函数 |

## 依赖

- numpy
- abacustest
- glob (标准库)
- itertools (标准库)