grn.infer Function Analysis
============================

This document provides a detailed explanation of the internal workflow of the ``grn.infer()`` function, including all sub-functions it calls and their purposes.

Function Overview
-----------------

``grn.infer()`` is the core function in the SpaGRN package for inferring Gene Regulatory Networks (GRNs) from spatially resolved transcriptomics data. This function implements a complete GRN inference pipeline consisting of 8 main steps.

Example Usage
~~~~~~~~~~~~~

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

Sub-functions Explained
------------------------

The grn.infer function calls the following sub-functions sequentially to complete the GRN inference pipeline:

Step 0: Initialization and Preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Set up output directories, temporary file directories, and prepare basic data structures

- Create output and temporary file directories
- Convert expression matrix format: ``exp_mat = self._data.to_df()``
- Set number of workers: if not specified, use ``cpu_count()`` to get CPU cores

Step 1: load_tfs()
~~~~~~~~~~~~~~~~~~

**Function**: ``self.load_tfs(tfs_fn)``

**Purpose**: Load the list of Transcription Factors (TFs)

**Details**:
    - Read predefined TF list from a text file
    - These TFs will serve as regulators in the gene regulatory network
    - If ``tfs_fn`` is None, use all genes as potential TFs

**Input**: Path to TF list file (e.g., 'mouse_TFs.txt')

**Output**: List of TF gene names

Step 2: load_database()
~~~~~~~~~~~~~~~~~~~~~~~~

**Function**: ``self.load_database(databases)``

**Purpose**: Load motif ranking databases

**Details**:
    - Load genome-wide ranking databases for motif enrichment analysis
    - Uses feather format database files (e.g., 'mouse.feather')
    - Database contains ranking information of genes for different motifs
    - Used in subsequent cisTarget analysis step

**Input**: Database file path or path pattern (supports wildcards)

**Output**: List of RankingDatabase objects

Step 3: spg() - Spatial-Proximity-Graph Model Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Function**: ``self.spg(...)``

**Purpose**: Infer TF-gene co-expression relationships using Spatial-Proximity-Graph (SPG) model

**Details**:
    This is the core step of GRN inference, consisting of multiple sub-steps:

    **3.1 Gene Selection (if gene_list not provided)**:
        
        a. Use Hotspot to calculate spatial autocorrelation of genes:
            - ``hotspot.Hotspot.create_knn_graph()``: Create K-nearest neighbors graph
            - ``hotspot.Hotspot.compute_autocorrelations()``: Compute Hotspot spatial autocorrelation
        
        b. Call ``spatial_autocorrelation()`` to compute multiple spatial statistics:
            - ``morans_i_p_values()``: Calculate Moran's I statistic and p-values
            - ``gearys_c()``: Calculate Geary's C statistic and p-values
            - ``getis_g()``: Calculate Getis-Ord G* statistic and p-values
            - ``fdr()``: Apply FDR correction to p-values from each method
        
        c. Call ``select_genes()`` to select Spatially Variable Genes (SVGs):
            - Based on ``methods`` parameter, choose which statistical methods' FDR values to use
            - Based on ``operation`` parameter ('intersection' or 'union'), combine results from multiple methods
            - Filter significant SVGs with FDR < 0.05

    **3.2 Compute Spatial Weight Matrix**:
        - ``neighbors_and_weights()``: Calculate proximity relationships and weights between cells
        - ``get_w()``: Generate spatial weight matrix W
        - ``flat_weights()``: Flatten weight matrix into computation-ready format

    **3.3 Compute TF-Gene Co-expression** (based on mode parameter):
        
        - **mode='moran'** (default, recommended):
            ``global_bivariate_moran_R()``: Calculate bivariate Moran's I statistic, measuring spatial co-expression between TF and target genes
        
        - **mode='geary'**:
            ``global_bivariate_gearys_C()``: Calculate bivariate Geary's C statistic
        
        - **mode='zscore'**:
            ``hs.compute_local_correlations()``: Use Hotspot to compute local correlation z-scores

**Input**: 
    - AnnData object
    - TF list
    - Spatial coordinates (latent_obsm_key='spatial')
    - Number of neighbors (n_neighbors=10)
    - Mode (mode='geary')
    - Statistical methods (methods=['FDR_I','FDR_C','FDR_G'])

**Output**: Adjacency matrix with three columns: TF, target, importance

**Saved File**: ``{mode}_adj.csv`` (e.g., geary_adj.csv)

Step 4: get_modules()
~~~~~~~~~~~~~~~~~~~~~

**Function**: ``self.get_modules(adjacencies, exp_mat, ...)``

**Purpose**: Create co-expression modules from adjacency matrix

**Details**:
    - Call pySCENIC's ``modules_from_adjacencies()`` function
    - Organize TF-target gene pairs into modules
    - Each module contains one TF and its associated target genes
    - Validate gene correlations within modules using expression matrix
    - ``rho_mask_dropouts``: Whether to mask zero values when computing correlations

**Input**: 
    - Adjacency matrix (TF-target-importance)
    - Expression matrix

**Output**: List of Regulon objects (preliminary modules, not yet motif-validated)

**Saved File**: ``modules.pkl``

Step 5: prune_modules() - cisTarget Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Function**: ``self.prune_modules(modules, dbs, motif_anno_fn, ...)``

**Purpose**: Refine modules through motif enrichment analysis to predict true regulons

**Details**:
    This is a critical quality control step based on pySCENIC cisTarget method:

    **5.1 Motif Enrichment Analysis**:
        - ``prune2df()``: Perform motif enrichment analysis for each module
        - Check if promoter regions of module target genes are enriched for TF binding motifs
        - Calculate NES (Normalized Enrichment Score) and AUC values
        - Keep only significantly enriched motifs (NES > nes_threshold)

    **5.2 Create Regulons**:
        - ``df2regulons()``: Create regulon objects from motif enrichment results
        - Keep only TF-target relationships supported by motifs
        - This step greatly improves reliability of inferred regulatory relationships

    **5.3 Process Results**:
        - ``get_regulon_dict()``: Convert regulon list to dictionary format {TF: [targets]}
        - Save to ``self.data.uns['regulon_dict']``

**Parameter Description**:
    - ``rank_threshold``: Number of ranked genes to consider (default: 1500)
    - ``auc_threshold``: AUC calculation threshold (default: 0.05)
    - ``nes_threshold``: NES threshold for filtering significant motifs (default: 3.0)
    - ``motif_similarity_fdr``: Motif similarity FDR threshold (default: 0.05)

**Input**: 
    - Module list
    - Ranking databases
    - Motif annotation file (e.g., 'mouse.tbl')

**Output**: List of motif-validated Regulon objects

**Saved Files**: 
    - ``motifs.csv``: Detailed motif enrichment analysis results
    - ``regulons.json``: Regulon dictionary

Step 6.0: cal_auc() - Cellular Enrichment Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Function**: ``self.cal_auc(exp_mat, regulons, ...)``

**Purpose**: Calculate activity of each regulon in each cell/spot using AUCell algorithm

**Details**:
    - Call pySCENIC's ``aucell()`` function
    - For each cell, calculate enrichment score (AUC value) for each regulon
    - AUC value reflects overall expression level of regulon target genes in that cell
    - Option to use gene weights (noweights parameter)
    - Option to normalize AUC values (normalize parameter)

**Input**: 
    - Expression matrix
    - Regulon list
    - AUC threshold (default: 0.05)
    - Whether to use weights (noweights)
    - Whether to normalize (normalize)

**Output**: AUC matrix (cells × regulons)

**Saved Location**: 
    - ``self.data.obsm['auc_mtx']``
    - File: ``auc_mtx.csv``

Step 6.1: Receptor Analysis (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: If niche_df (ligand-receptor database) is provided, perform receptor-related analysis

**Sub-steps**:

    **6.1.1 get_filtered_receptors()**:
        - ``get_filtered_genes()``: Identify genes filtered out by cisTarget
        - ``intersection_ci()``: Find receptor genes among these genes
        - Find receptor genes associated with each TF
        - Save to ``self.receptor_dict``

    **6.1.2 receptor_auc()**:
        - Calculate AUC values for receptor gene modules
        - Use ``aucell()`` function
        - Return receptor AUC matrix

    **6.1.3 isr()**:
        - Calculate Integrated Signaling Receptor (ISR) matrix
        - Combine regulon AUC and receptor AUC
        - Sum values for regulons with the same name
        - Save to ``self.data.obsm['isr']``

**Input**: 
    - Ligand-receptor database (niche_df)
    - Receptor column name (receptor_key='to')

**Output**: 
    - Receptor dictionary
    - ISR matrix

Step 7: cal_regulon_score()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Function**: ``self.cal_regulon_score(cluster_label=cluster_label, ...)``

**Purpose**: Calculate Regulon Specificity Scores (RSS)

**Details**:
    - Call pySCENIC's ``regulon_specificity_scores()`` function
    - Calculate specificity of each regulon in each cell type
    - Higher RSS value indicates the regulon is more specific to that cell type
    - Based on Jensen-Shannon divergence
    - Used to identify cell type-specific regulatory programs

**Input**: 
    - AUC matrix (from Step 6.0)
    - Cell type labels (cluster_label='celltype')

**Output**: RSS matrix (regulons × cell types)

**Saved Location**: 
    - ``self.data.uns['rss']``
    - File: ``regulon_specificity_scores.txt``

Step 8: Save Results
~~~~~~~~~~~~~~~~~~~~

**Purpose**: Save all results to h5ad file

**Details**:
    - Use ``self.data.write_h5ad()`` to save complete AnnData object
    - File contains:
        - Original expression data
        - Adjacency matrix (uns['adj'])
        - Regulon dictionary (uns['regulon_dict'])
        - AUC matrix (obsm['auc_mtx'])
        - RSS matrix (uns['rss'])
        - ISR matrix (if computed, obsm['isr'])
        - Receptor dictionary (if computed, uns['receptor_dict'])

**Output File**: ``{project_name}_spagrn.h5ad``

Complete Workflow Summary
--------------------------

.. code-block:: text

    Input: AnnData object + parameter configuration
        ↓
    Step 1: Load TF list
        ↓
    Step 2: Load Motif database
        ↓
    Step 3: SPG model inference
        ├── 3.1: Select Spatially Variable Genes (SVGs)
        │   ├── Hotspot analysis
        │   ├── Moran's I analysis
        │   ├── Geary's C analysis
        │   ├── Getis-Ord G analysis
        │   └── FDR correction and gene selection
        ├── 3.2: Compute spatial weight matrix
        └── 3.3: Compute TF-gene co-expression
            └── Output: Adjacency matrix (TF-target-importance)
        ↓
    Step 4: Create co-expression modules
        └── Output: Preliminary modules (Modules)
        ↓
    Step 5: cisTarget Motif enrichment analysis
        ├── Motif enrichment calculation
        ├── Filter low-quality relationships
        └── Output: Refined Regulons
        ↓
    Step 6.0: AUCell cellular enrichment analysis
        └── Output: AUC matrix (cells × regulons)
        ↓
    Step 6.1: Receptor analysis (optional)
        ├── Identify receptor genes
        ├── Calculate receptor AUC
        └── Output: ISR matrix
        ↓
    Step 7: Calculate Regulon Specificity Scores
        └── Output: RSS matrix (regulons × cell types)
        ↓
    Step 8: Save results
        └── Output: {project_name}_spagrn.h5ad

Key Parameters
--------------

Spatial Analysis Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **latent_obsm_key**: Key for spatial coordinates in adata.obsm, default 'spatial'
- **n_neighbors**: K-nearest neighbors count, controls spatial neighborhood size, default 10
- **model**: Null model for gene expression, options:
    - 'danb': Depth-Adjusted Negative Binomial (recommended for UMI data)
    - 'bernoulli': Detection probability model
    - 'normal': Depth-Adjusted Normal
- **mode**: Co-expression calculation mode:
    - 'moran': Bivariate Moran's I (recommended)
    - 'geary': Bivariate Geary's C
    - 'zscore': Hotspot z-score

Gene Selection Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **gene_list**: Specified gene list, if provided, skips automatic gene selection
- **methods**: List of FDR methods for gene selection, e.g., ['FDR_I','FDR_C','FDR_G']
    - 'FDR_I': FDR-corrected p-values from Moran's I
    - 'FDR_C': FDR-corrected p-values from Geary's C
    - 'FDR_G': FDR-corrected p-values from Getis-Ord G
- **operation**: Method to combine multiple methods:
    - 'intersection': Take intersection (more stringent)
    - 'union': Take union (more permissive)

Performance Parameters
~~~~~~~~~~~~~~~~~~~~~~

- **num_workers**: Number of parallel worker processes, default uses all CPU cores
- **cache**: Whether to use cached intermediate results, speeds up repeated runs
- **save_tmp**: Whether to save intermediate result files

Quality Control Parameters (set via grn.params)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **rank_threshold**: Motif ranking threshold, default 1500
- **prune_auc_threshold**: Motif AUC threshold, default 0.05
- **nes_threshold**: Motif NES threshold, default 3.0
- **motif_similarity_fdr**: Motif similarity FDR, default 0.05
- **auc_threshold**: AUCell AUC threshold, default 0.05

Output Files
------------

Main Output
~~~~~~~~~~~

- **{project_name}_spagrn.h5ad**: AnnData file containing all results

Intermediate Files (if save_tmp=True)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **{mode}_adj.csv**: TF-target gene adjacency matrix
- **more_stats.csv**: Spatial autocorrelation statistics results
- **selected_genes.txt**: List of selected spatially variable genes
- **modules.pkl**: Preliminary co-expression modules
- **motifs.csv**: Detailed motif enrichment analysis results
- **regulons.json**: Regulon dictionary
- **auc_mtx.csv**: AUC matrix
- **regulon_specificity_scores.txt**: RSS scores

Usage Recommendations
---------------------

1. **First run**: Set ``cache=False, save_tmp=True`` to save all intermediate results
2. **Debugging and optimization**: Set ``cache=True`` to reuse computed results and quickly test different parameters
3. **Number of neighbors**: Use ``n_neighbors=10`` for high-resolution data, can increase to 30 for low-resolution data
4. **Gene selection**: Recommend using ``methods=['FDR_I','FDR_C','FDR_G'], operation='intersection'`` for high-quality SVGs
5. **Co-expression mode**: Recommend using ``mode='moran'`` for better detection of regulatory networks in rare cell types
6. **Ligand-receptor analysis**: If studying cell communication, provide ``niche_df`` parameter

References
----------

- SpaGRN paper: https://www.biorxiv.org/content/10.1101/2023.01.01.522397v1
- pySCENIC documentation: https://pyscenic.readthedocs.io/
- Hotspot documentation: https://hotspot.readthedocs.io/

Related Functions
-----------------

For detailed implementation of each sub-function, please refer to:

- ``spatial_autocorrelation()``: src/spagrn/regulatory_network.py, lines 374-438
- ``spg()``: src/spagrn/regulatory_network.py, lines 526-674
- ``get_modules()``: src/spagrn/regulatory_network.py, lines 679-710
- ``prune_modules()``: src/spagrn/regulatory_network.py, lines 715-777
- ``cal_auc()``: src/spagrn/regulatory_network.py, lines 782-834
- ``cal_regulon_score()``: src/spagrn/network.py, lines 321-334
