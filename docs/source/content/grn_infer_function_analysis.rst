grn.infer 函数分析
====================

本文档详细介绍 ``grn.infer()`` 函数的内部工作流程，包括它调用的所有子函数及其功能。

函数概述
--------

``grn.infer()`` 是 SpaGRN 包中的核心函数，用于从空间转录组数据推断基因调控网络 (Gene Regulatory Networks, GRNs)。该函数实现了一个完整的 GRN 推断流程，包括 8 个主要步骤。

调用示例
~~~~~~~~

.. code-block:: python

    grn.infer(database_fn,
              motif_anno_fn,
              tfs_fn,
              niche_df=niches,
              gene_list=None,
              num_workers=15,
              cache=True,
              output_dir=output_path,
              save_tmp=True,
              layers='count',
              latent_obsm_key='spatial',
              model='danb',
              n_neighbors=10,
              methods=['FDR_I','FDR_C','FDR_G'],
              operation='intersection',
              mode='geary',
              cluster_label='celltype')

内部子函数详解
--------------

grn.infer 函数按顺序调用以下子函数来完成 GRN 推断流程：

步骤 0: 初始化和准备
~~~~~~~~~~~~~~~~~~~~

**功能**: 设置输出目录、临时文件目录，准备基础数据结构

- 创建输出目录和临时文件目录
- 转换表达矩阵格式: ``exp_mat = self._data.to_df()``
- 设置工作线程数: 如果未指定，使用 ``cpu_count()`` 获取 CPU 核心数

步骤 1: load_tfs()
~~~~~~~~~~~~~~~~~~

**函数**: ``self.load_tfs(tfs_fn)``

**功能**: 加载转录因子 (Transcription Factors, TFs) 列表

**详细说明**:
    - 从文本文件中读取预定义的转录因子列表
    - 这些转录因子将作为基因调控网络的调控因子
    - 如果 ``tfs_fn`` 为 None，则使用所有基因作为潜在 TF

**输入**: TF 列表文件路径 (例如: 'mouse_TFs.txt')

**输出**: TF 基因名称列表

步骤 2: load_database()
~~~~~~~~~~~~~~~~~~~~~~~

**函数**: ``self.load_database(databases)``

**功能**: 加载 motif 排名数据库 (ranking databases)

**详细说明**:
    - 加载用于 motif 富集分析的基因组排名数据库
    - 使用 feather 格式的数据库文件 (例如: 'mouse.feather')
    - 数据库包含基因在不同 motif 中的排名信息
    - 用于后续的 cisTarget 分析步骤

**输入**: 数据库文件路径或路径模式 (支持通配符)

**输出**: RankingDatabase 对象列表

步骤 3: spg() - 空间接近图模型推断
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**函数**: ``self.spg(...)``

**功能**: 使用空间接近图 (Spatial-Proximity-Graph, SPG) 模型推断 TF-基因共表达关系

**详细说明**:
    这是 GRN 推断的核心步骤，包含多个子步骤:

    **3.1 基因选择 (如果未提供 gene_list)**:
        
        a. 使用 Hotspot 计算基因的空间自相关性:
            - ``hotspot.Hotspot.create_knn_graph()``: 创建 K 近邻图
            - ``hotspot.Hotspot.compute_autocorrelations()``: 计算 Hotspot 空间自相关
        
        b. 调用 ``spatial_autocorrelation()`` 计算多种空间统计指标:
            - ``morans_i_p_values()``: 计算 Moran's I 统计量及 p 值
            - ``gearys_c()``: 计算 Geary's C 统计量及 p 值  
            - ``getis_g()``: 计算 Getis-Ord G* 统计量及 p 值
            - ``fdr()``: 对每种方法的 p 值进行 FDR 校正
        
        c. 调用 ``select_genes()`` 选择空间可变基因 (Spatially Variable Genes, SVGs):
            - 根据 ``methods`` 参数选择使用哪些统计方法的 FDR 值
            - 根据 ``operation`` 参数 ('intersection' 或 'union') 组合多个方法的结果
            - 筛选 FDR < 0.05 的显著空间可变基因

    **3.2 计算空间权重矩阵**:
        - ``neighbors_and_weights()``: 计算细胞间的邻近关系和权重
        - ``get_w()``: 生成空间权重矩阵 W
        - ``flat_weights()``: 将权重矩阵展平为适合计算的格式

    **3.3 计算 TF-基因共表达关系** (根据 mode 参数):
        
        - **mode='moran'** (默认推荐):
            ``global_bivariate_moran_R()``: 计算双变量 Moran's I 统计量，衡量 TF 与靶基因的空间共表达
        
        - **mode='geary'**:
            ``global_bivariate_gearys_C()``: 计算双变量 Geary's C 统计量
        
        - **mode='zscore'**:
            ``hs.compute_local_correlations()``: 使用 Hotspot 计算局部相关性 z-score

**输入**: 
    - AnnData 对象
    - TF 列表
    - 空间坐标 (latent_obsm_key='spatial')
    - 邻居数量 (n_neighbors=10)
    - 模式 (mode='geary')
    - 统计方法 (methods=['FDR_I','FDR_C','FDR_G'])

**输出**: 邻接矩阵 (adjacencies)，包含三列: TF, target, importance

**保存文件**: ``{mode}_adj.csv`` (如: geary_adj.csv)

步骤 4: get_modules()
~~~~~~~~~~~~~~~~~~~~~

**函数**: ``self.get_modules(adjacencies, exp_mat, ...)``

**功能**: 从邻接矩阵创建共表达模块 (co-expression modules)

**详细说明**:
    - 调用 pySCENIC 的 ``modules_from_adjacencies()`` 函数
    - 将 TF-靶基因对组织成模块
    - 每个模块包含一个 TF 及其相关的靶基因
    - 使用表达矩阵验证模块中基因的相关性
    - ``rho_mask_dropouts``: 是否在计算相关性时屏蔽零值

**输入**: 
    - 邻接矩阵 (TF-target-importance)
    - 表达矩阵

**输出**: Regulon 对象列表 (初步模块，未经过 motif 验证)

**保存文件**: ``modules.pkl``

步骤 5: prune_modules() - cisTarget 分析
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**函数**: ``self.prune_modules(modules, dbs, motif_anno_fn, ...)``

**功能**: 通过 motif 富集分析精炼模块，预测真实的调控子 (regulons)

**详细说明**:
    这是基于 pySCENIC cisTarget 方法的关键质控步骤:

    **5.1 Motif 富集分析**:
        - ``prune2df()``: 对每个模块进行 motif 富集分析
        - 检查模块的靶基因启动子区域是否富集 TF 结合 motif
        - 计算 NES (Normalized Enrichment Score) 和 AUC 值
        - 只保留显著富集的 motif (NES > nes_threshold)

    **5.2 创建 Regulons**:
        - ``df2regulons()``: 从 motif 富集结果创建 regulon 对象
        - 只保留有 motif 支持的 TF-靶基因关系
        - 这一步大幅提高了推断调控关系的可靠性

    **5.3 结果处理**:
        - ``get_regulon_dict()``: 将 regulon 列表转换为字典格式 {TF: [targets]}
        - 保存到 ``self.data.uns['regulon_dict']``

**参数说明**:
    - ``rank_threshold``: 考虑的排名基因数量 (默认: 1500)
    - ``auc_threshold``: AUC 计算阈值 (默认: 0.05)
    - ``nes_threshold``: NES 阈值，用于筛选显著 motif (默认: 3.0)
    - ``motif_similarity_fdr``: Motif 相似性 FDR 阈值 (默认: 0.05)

**输入**: 
    - 模块列表
    - Ranking 数据库
    - Motif 注释文件 (例如: 'mouse.tbl')

**输出**: 经过 motif 验证的 Regulon 对象列表

**保存文件**: 
    - ``motifs.csv``: motif 富集分析详细结果
    - ``regulons.json``: regulon 字典

步骤 6.0: cal_auc() - 细胞富集分析
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**函数**: ``self.cal_auc(exp_mat, regulons, ...)``

**功能**: 使用 AUCell 算法计算每个 regulon 在每个细胞/spot 中的活性

**详细说明**:
    - 调用 pySCENIC 的 ``aucell()`` 函数
    - 对每个细胞，计算每个 regulon 的富集得分 (AUC 值)
    - AUC 值反映了 regulon 靶基因在该细胞中的整体表达水平
    - 可选择是否使用基因权重 (noweights 参数)
    - 可选择是否归一化 AUC 值 (normalize 参数)

**输入**: 
    - 表达矩阵
    - Regulon 列表
    - AUC 阈值 (默认: 0.05)
    - 是否使用权重 (noweights)
    - 是否归一化 (normalize)

**输出**: AUC 矩阵 (细胞 × regulons)

**保存位置**: 
    - ``self.data.obsm['auc_mtx']``
    - 文件: ``auc_mtx.csv``

步骤 6.1: 受体分析 (可选)
~~~~~~~~~~~~~~~~~~~~~~~~~

**功能**: 如果提供了 niche_df (配体-受体数据库)，进行受体相关分析

**子步骤**:

    **6.1.1 get_filtered_receptors()**:
        - ``get_filtered_genes()``: 识别被 cisTarget 过滤掉的基因
        - ``intersection_ci()``: 查找这些基因中的受体基因
        - 为每个 TF 找到相关的受体基因
        - 保存到 ``self.receptor_dict``

    **6.1.2 receptor_auc()**:
        - 为受体基因模块计算 AUC 值
        - 使用 ``aucell()`` 函数
        - 返回受体 AUC 矩阵

    **6.1.3 isr()**:
        - 计算整合信号受体 (Integrated Signaling Receptor, ISR) 矩阵
        - 合并 regulon AUC 和受体 AUC
        - 对同名 regulon 求和
        - 保存到 ``self.data.obsm['isr']``

**输入**: 
    - 配体-受体数据库 (niche_df)
    - 受体列名 (receptor_key='to')

**输出**: 
    - 受体字典
    - ISR 矩阵

步骤 7: cal_regulon_score()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**函数**: ``self.cal_regulon_score(cluster_label=cluster_label, ...)``

**功能**: 计算 regulon 特异性得分 (Regulon Specificity Scores, RSS)

**详细说明**:
    - 调用 pySCENIC 的 ``regulon_specificity_scores()`` 函数
    - 计算每个 regulon 在每种细胞类型中的特异性
    - RSS 值越高，表明该 regulon 在该细胞类型中越特异
    - 基于 Jensen-Shannon 散度计算
    - 用于识别细胞类型特异的调控程序

**输入**: 
    - AUC 矩阵 (来自步骤 6.0)
    - 细胞类型标签 (cluster_label='celltype')

**输出**: RSS 矩阵 (regulons × 细胞类型)

**保存位置**: 
    - ``self.data.uns['rss']``
    - 文件: ``regulon_specificity_scores.txt``

步骤 8: 保存结果
~~~~~~~~~~~~~~~~

**功能**: 将所有结果保存到 h5ad 文件

**详细说明**:
    - 使用 ``self.data.write_h5ad()`` 保存完整的 AnnData 对象
    - 文件包含:
        - 原始表达数据
        - 邻接矩阵 (uns['adj'])
        - Regulon 字典 (uns['regulon_dict'])
        - AUC 矩阵 (obsm['auc_mtx'])
        - RSS 矩阵 (uns['rss'])
        - ISR 矩阵 (如果计算了，obsm['isr'])
        - 受体字典 (如果计算了，uns['receptor_dict'])

**输出文件**: ``{project_name}_spagrn.h5ad``

完整工作流程总结
----------------

.. code-block:: text

    输入: AnnData 对象 + 参数配置
        ↓
    步骤 1: 加载 TF 列表
        ↓
    步骤 2: 加载 Motif 数据库
        ↓
    步骤 3: SPG 模型推断
        ├── 3.1: 选择空间可变基因 (SVGs)
        │   ├── Hotspot 分析
        │   ├── Moran's I 分析
        │   ├── Geary's C 分析
        │   ├── Getis-Ord G 分析
        │   └── FDR 校正和基因选择
        ├── 3.2: 计算空间权重矩阵
        └── 3.3: 计算 TF-基因共表达
            └── 输出: 邻接矩阵 (TF-target-importance)
        ↓
    步骤 4: 创建共表达模块
        └── 输出: 初步模块 (Modules)
        ↓
    步骤 5: cisTarget Motif 富集分析
        ├── Motif 富集计算
        ├── 过滤低质量关系
        └── 输出: 精炼的 Regulons
        ↓
    步骤 6.0: AUCell 细胞富集分析
        └── 输出: AUC 矩阵 (细胞 × regulons)
        ↓
    步骤 6.1: 受体分析 (可选)
        ├── 识别受体基因
        ├── 计算受体 AUC
        └── 输出: ISR 矩阵
        ↓
    步骤 7: 计算 Regulon 特异性得分
        └── 输出: RSS 矩阵 (regulons × 细胞类型)
        ↓
    步骤 8: 保存结果
        └── 输出: {project_name}_spagrn.h5ad

关键参数说明
------------

空间分析参数
~~~~~~~~~~~~

- **latent_obsm_key**: 空间坐标在 adata.obsm 中的键，默认 'spatial'
- **n_neighbors**: K 近邻数量，控制空间邻域大小，默认 10
- **model**: 基因表达的零模型，可选:
    - 'danb': 深度调整负二项分布 (推荐用于 UMI 数据)
    - 'bernoulli': 检测概率模型
    - 'normal': 深度调整正态分布
- **mode**: 共表达计算模式:
    - 'moran': 双变量 Moran's I (推荐)
    - 'geary': 双变量 Geary's C
    - 'zscore': Hotspot z-score

基因选择参数
~~~~~~~~~~~~

- **gene_list**: 指定基因列表，如果提供则跳过自动基因选择
- **methods**: 用于基因选择的 FDR 方法列表，例如 ['FDR_I','FDR_C','FDR_G']
    - 'FDR_I': Moran's I 的 FDR 校正 p 值
    - 'FDR_C': Geary's C 的 FDR 校正 p 值
    - 'FDR_G': Getis-Ord G 的 FDR 校正 p 值
- **operation**: 组合多个方法的方式:
    - 'intersection': 取交集（更严格）
    - 'union': 取并集（更宽松）

性能参数
~~~~~~~~

- **num_workers**: 并行计算的工作进程数，默认使用所有 CPU 核心
- **cache**: 是否使用缓存的中间结果，加速重复运行
- **save_tmp**: 是否保存中间结果文件

质量控制参数 (通过 grn.params 设置)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **rank_threshold**: Motif 排名阈值，默认 1500
- **prune_auc_threshold**: Motif AUC 阈值，默认 0.05
- **nes_threshold**: Motif NES 阈值，默认 3.0
- **motif_similarity_fdr**: Motif 相似性 FDR，默认 0.05
- **auc_threshold**: AUCell AUC 阈值，默认 0.05

输出文件
--------

主要输出
~~~~~~~~

- **{project_name}_spagrn.h5ad**: 包含所有结果的 AnnData 文件

中间文件 (如果 save_tmp=True)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **{mode}_adj.csv**: TF-靶基因邻接矩阵
- **more_stats.csv**: 空间自相关统计结果
- **selected_genes.txt**: 选中的空间可变基因列表
- **modules.pkl**: 初步共表达模块
- **motifs.csv**: Motif 富集分析详细结果
- **regulons.json**: Regulon 字典
- **auc_mtx.csv**: AUC 矩阵
- **regulon_specificity_scores.txt**: RSS 得分

使用建议
--------

1. **首次运行**: 设置 ``cache=False, save_tmp=True`` 保存所有中间结果
2. **调试和优化**: 设置 ``cache=True`` 重用计算结果，快速测试不同参数
3. **邻居数量**: 对于高分辨率数据使用 ``n_neighbors=10``，低分辨率数据可增加到 30
4. **基因选择**: 推荐使用 ``methods=['FDR_I','FDR_C','FDR_G'], operation='intersection'`` 获得高质量 SVGs
5. **共表达模式**: 推荐使用 ``mode='moran'`` 以更好地检测罕见细胞类型的调控网络
6. **配体-受体分析**: 如需研究细胞通讯，提供 ``niche_df`` 参数

参考资源
--------

- SpaGRN 论文: https://www.biorxiv.org/content/10.1101/2023.01.01.522397v1
- pySCENIC 文档: https://pyscenic.readthedocs.io/
- Hotspot 文档: https://hotspot.readthedocs.io/

相关函数
--------

详细了解各子函数的实现，请参考:

- ``spatial_autocorrelation()``: src/spagrn/regulatory_network.py, 第 374-438 行
- ``spg()``: src/spagrn/regulatory_network.py, 第 526-674 行
- ``get_modules()``: src/spagrn/regulatory_network.py, 第 679-710 行
- ``prune_modules()``: src/spagrn/regulatory_network.py, 第 715-777 行
- ``cal_auc()``: src/spagrn/regulatory_network.py, 第 782-834 行
- ``cal_regulon_score()``: src/spagrn/network.py, 第 321-334 行
