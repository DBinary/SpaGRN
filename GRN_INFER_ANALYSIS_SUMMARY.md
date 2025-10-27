# grn.infer 函数分析总结

根据对 SpaGRN 代码的详细分析，`grn.infer()` 函数内部调用了以下子函数：

## 主要执行步骤

### 步骤 1: load_tfs(tfs_fn)
**功能**: 从文本文件加载转录因子(TF)列表

### 步骤 2: load_database(databases)
**功能**: 加载 motif 排名数据库，用于后续的 motif 富集分析

### 步骤 3: spg() - 空间接近图模型推断
**功能**: 这是核心的 GRN 推断步骤，内部包含多个子步骤：

#### 3.1 基因选择（当未提供 gene_list 时）
- `hotspot.Hotspot.create_knn_graph()`: 创建 K 近邻图
- `hotspot.Hotspot.compute_autocorrelations()`: 计算 Hotspot 空间自相关
- `spatial_autocorrelation()`: 计算多种空间统计指标
  - `morans_i_p_values()`: 计算 Moran's I 统计量
  - `gearys_c()`: 计算 Geary's C 统计量
  - `getis_g()`: 计算 Getis-Ord G* 统计量
  - `fdr()`: FDR 校正
- `select_genes()`: 选择空间可变基因(SVGs)

#### 3.2 计算空间权重矩阵
- `neighbors_and_weights()`: 计算细胞间的邻近关系和权重
- `get_w()`: 生成空间权重矩阵
- `flat_weights()`: 展平权重矩阵

#### 3.3 计算 TF-基因共表达关系
根据 `mode` 参数选择不同的方法：
- **mode='moran'**: `global_bivariate_moran_R()` - 双变量 Moran's I
- **mode='geary'**: `global_bivariate_gearys_C()` - 双变量 Geary's C  
- **mode='zscore'**: `hs.compute_local_correlations()` - Hotspot z-score

**输出**: 邻接矩阵(adjacencies)，包含 TF-target-importance 三列

### 步骤 4: get_modules(adjacencies, exp_mat)
**功能**: 从邻接矩阵创建共表达模块
- 调用 pySCENIC 的 `modules_from_adjacencies()` 函数
- 将 TF-靶基因对组织成模块

**输出**: 初步的 Regulon 对象列表（模块）

### 步骤 5: prune_modules() - cisTarget 分析
**功能**: 通过 motif 富集分析精炼模块，预测真实的调控子
- `prune2df()`: 对每个模块进行 motif 富集分析
- `df2regulons()`: 从 motif 富集结果创建 regulon 对象
- `get_regulon_dict()`: 将 regulon 列表转换为字典格式

**输出**: 经过 motif 验证的 Regulon 对象列表

### 步骤 6.0: cal_auc(exp_mat, regulons)
**功能**: 使用 AUCell 算法计算每个 regulon 在每个细胞中的活性
- 调用 pySCENIC 的 `aucell()` 函数
- 计算 AUC 值（Area Under Curve）

**输出**: AUC 矩阵（细胞 × regulons）

### 步骤 6.1: 受体分析（可选，当提供 niche_df 时）
**功能**: 分析受体基因与调控网络的关系
- `get_filtered_receptors()`: 识别受体基因
  - `get_filtered_genes()`: 获取被 cisTarget 过滤的基因
  - `intersection_ci()`: 在过滤基因中查找受体
- `receptor_auc()`: 计算受体基因模块的 AUC 值
- `isr()`: 计算整合信号受体(ISR)矩阵

**输出**: ISR 矩阵

### 步骤 7: cal_regulon_score(cluster_label)
**功能**: 计算 regulon 特异性得分(RSS)
- 调用 pySCENIC 的 `regulon_specificity_scores()` 函数
- 评估每个 regulon 在不同细胞类型中的特异性

**输出**: RSS 矩阵（regulons × 细胞类型）

### 步骤 8: write_h5ad()
**功能**: 保存所有结果到 h5ad 文件
- 保存表达数据、邻接矩阵、regulon 字典、AUC 矩阵、RSS 矩阵等

**输出**: {project_name}_spagrn.h5ad 文件

## 完整流程图

```
输入数据 (AnnData)
    ↓
1. 加载 TF 列表 (load_tfs)
    ↓
2. 加载 Motif 数据库 (load_database)
    ↓
3. SPG 模型推断 (spg)
    ├─ 3.1 选择空间可变基因
    │   ├─ Hotspot 分析
    │   ├─ Moran's I / Geary's C / Getis-Ord G 分析
    │   └─ FDR 校正和基因选择
    ├─ 3.2 计算空间权重矩阵
    └─ 3.3 计算 TF-基因共表达
    ↓
    邻接矩阵
    ↓
4. 创建共表达模块 (get_modules)
    ↓
    初步模块
    ↓
5. cisTarget Motif 富集分析 (prune_modules)
    ↓
    精炼的 Regulons
    ↓
6.0 AUCell 细胞富集分析 (cal_auc)
    ↓
    AUC 矩阵
    ↓
6.1 受体分析 (可选)
    ↓
    ISR 矩阵
    ↓
7. 计算 Regulon 特异性得分 (cal_regulon_score)
    ↓
    RSS 矩阵
    ↓
8. 保存结果 (write_h5ad)
    ↓
输出: {project_name}_spagrn.h5ad
```

## 关键参数说明

根据您的调用示例：

- **database_fn**: motif 排名数据库文件
- **motif_anno_fn**: motif 注释文件
- **tfs_fn**: TF 列表文件
- **niche_df=niches**: 配体-受体数据库（用于受体分析）
- **num_workers=15**: 并行计算的工作进程数
- **cache=True**: 使用缓存的中间结果
- **layers='count'**: 使用的数据层
- **model='danb'**: 深度调整负二项分布模型
- **n_neighbors=10**: K 近邻数量
- **methods=['FDR_I','FDR_C','FDR_G']**: 使用三种空间自相关方法的 FDR 值
- **operation='intersection'**: 取交集选择基因（更严格）
- **mode='geary'**: 使用 Geary's C 计算共表达
- **cluster_label='celltype'**: 细胞类型标签列名

## 详细文档位置

完整的函数分析文档已添加到：
- 中文版: `docs/source/content/grn_infer_function_analysis.rst`
- 英文版: `docs/source/content/grn_infer_function_analysis_en.rst`
