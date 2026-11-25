# ECE592_Final_Project

This project outlines the code created for the Final Project in ECE592. final-project.ipynb contains all the code used for this project. The data I used was collected by the Gemma Lab at NC State and can be provided upon request. I ran my code through Kaggle. 

This project is for a message-passing graph neural network (GNN) that predicts transcription factor (TF) - gene target edges in maize from expression across 214 B73 × Palomero Toluqueño recombinant inbred lines (RILs), using the mGRN+ leaf network as the graph prior. Each edge carries a genotype × phenotype interaction feature calculated from a regression. A differential score Δ, represents the change in predicted edge probability between low and high cold-tolerant phenotype scores. This is tested on data from the Gemma lab, and on SERGIO-simulated data with known differential edges.

## Files

`final-project.ipynb:` The Kaggle notebook used to make the results. 
`requirements.txt` | Python packages. *(Add if not present — see Setup.)*
`README.md`

**Data are not included** The pipeline takes these files as an input:
`counts`, `plant_counts_raw_UMI.csv`: Raw 3′ UMI counts, 375 plants × 15,346 genes
`meta`, `plant_metadata.csv`: Per-plant metadata, including TMM normalization factors
`pheno`, `Phenotyping_scores_manual_-_All_runs.csv`: Manual damage scores (anthocyanin, necrosis, wilt)
`mgrn`, `mGRN_leaf.csv`: mGRN+ leaf regulatory network (TF, Target; B73 v4 IDs)
`idmap`, `B73v4_to_B73v5.tsv`: MaizeGDB v4 to v5 gene ID conversion
`vcf` or `genotype_csv`, `b73_x_pt_rils_nils_dh.vcf(.gz)` or `genotype_dosage.csv`: genotype calls per sample. The CSV is derived from the VCF with `read_vcf_dosage`
`sample_key`,`sample_key.tsv`: Genotyped sample to line
`marker_map`,`marker_map.tsv`: Marker gene positions (v5)

## Setup

Python ≥ 3.10.

```bash
pip install numpy pandas scipy scikit-learn matplotlib torch
```

SERGIO (used only for the simulation, section 12) is **downloaded automatically** from GitHub the first time it runs (`setup_sergio`), and its outdated numpy calls (`np.int`, `np.float`) are patched on load. On Kaggle, turn the Internet on in settings.

---

## Running the experiments

### Locally

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
mkdir -p data/raw            # put the input files here
python gnn_diffgrn_pipeline.py
```

### On Kaggle

1. Attach the dataset(s) containing the input files (any folder names — files are found automatically by `resolve_paths`).
2. Turn on GPU and Internet.
3. Run all cells, preferably as a committed version (Save Version, then Save & Run All). Results are written to `/kaggle/working/results`.

### Execution of a Run

Running the file executes:

1. `run_all()` — for each of 3 random TF splits and both edge-feature modes (`genotype`, `tf_expression`): MLP baseline, full GNN and all ablations, and Δ with a 20-permutation null.
2. `summarize()` and `figures()` — summary tables and figures.
3. `run_simulation()` — SERGIO simulation with planted differential edges; scores Δ against the ground truth.

### Common settings (`CFG`)

```python
CFG["z_modes"] = ["genotype", "tf_expression"]   # edge-feature modes to run
CFG["conditions"] = ["composite"]                # phenotype(s); also "antho", "necro", "wilt"
CFG["seeds"] = [0, 1, 2]                         # TF splits
CFG["epochs"], CFG["patience"] = 400, 30
CFG["n_permutations"] = 20                       # Delta null
CFG["sim"]["enabled"] = True                     # SERGIO simulation on/off
CFG["sim"]["seed"] = 0                           # simulated network
```

Quick test (minutes): `CFG["seeds"] = [0]; CFG["epochs"] = 5; CFG["n_permutations"] = 2`.

### Outputs (`results/`)

`linkpred_results.csv`, `linkpred_summary.csv`: Test AUROC / AUPRC / EPR per mode × variant × seed, and mean ± SD
`delta_summary.csv`, `delta_summary_by_mode.csv`: Δ significance counts, count above null 99th percentile, agreement with β₃
`delta_<mode>_<condition>_seed<k>.csv`: Δ, p-value and q-value for every edge 
`sim_results.csv`, `sim_summary.csv`: Simulation: link prediction and recovery of planted differential edges
`fig_linkpred.png`, `fig_delta_vs_null.png`: Report figures
`history_*.csv`, `run_log.txt`, `mgrn_mapped.csv`: Training curves, full log, final edge list

## Code organization

All functions are in `gnn_diffgrn_pipeline.py`, in the following sections.

### Configuration

| `CFG` | All settings: file names, filters, phenotype signs, modes, model and training hyperparameters, simulation settings |
| `log` | Prints a message and stores it for `run_log.txt` |
| `resolve_paths` | Finds every input file (all of `/kaggle/input` on Kaggle, `raw_dir` locally); sets `RAW` / `OUT` |

### Data preparation

`canonical_line`: harmonizes line names across files (e.g. `LANPT16B023` and `RIL023` → `RIL023`)
`parse_pheno_sample`: parses phenotype image names (`RUN1_A10_RIL181.tif` → run, row, column, line)
`open_maybe_gzip`: opens a text file whether or not it is gzip-compressed |
`read_vcf_dosage`: reads genotype calls into a samples × markers matrix of PT allele counts (0/1/2)
`load_genotypes`: loads genotypes (CSV or VCF), the sample key and the marker map
`verified_labels`: assigns each RNA sample its genotype-verified line (corrects mislabeled plate PN3; drops unverifiable samples)
`genotype_line_matrix`: lines × markers PT allele fraction, averaged over each line's genotyped plants
`assign_genes_to_markers`: maps every gene to itself (if a marker) or the nearest marker on its chromosome
`load_genotype_Z`: lines × genes genotype at each gene's locus
`prepare_data`: filters genes and plants, log-CPM normalization (TMM), line means of expression and phenotype; caches results

### Phenotype
`get_P`: tolerance score per line: z-scored, sign-aligned traits; composite = re-standardized mean
`tail_lines`: bottom / top fraction of lines by P (seeded tie-breaking) |
`contrast_values`: p_low / p_high = mean P of the bottom / top 10% of lines |

### Edge features (G × P regression)
`make_Z`: TF predictor: genotype at the TF locus, or the TF's standardized expression |
`fit_gxp`: Fits Xᵢ = β₀ + β₁Zⱼ + β₂P + β₃ZⱼP for every edge (one least-squares solve per TF); returns β₁, β₃, t(β₃)

### Graph
`load_network`: strict 1:1 v4→v5 ID mapping; keeps edges between expressed genes
`split_by_tf`: assigns whole TFs to train/val/test (70/10/20 of edges), stratified by out-degree
`sample_negatives`: target-corruption negatives (same TF, degree-weighted target; never a known edge)
`build_graph`: node features (expression + TF flag), query sets, message-passing edges (train only, with reverse copies)
`check_leakage`: stops the run if held-out edges reach message passing, negatives are real edges, or TFs overlap splits
`attach_betas`: fits Eq. 2 for all pairs; attaches coefficients and standardization constants
`e_of_p`: standardized edge feature e_ij(p) = β₁ + β₃(p − p̄)
`drop_untestable_tfs`: keeps TFs whose genotype varies in ≥ 30 lines; same edge set used for both modes

### 6. Model
| Name | Purpose |
|---|---|
| `mean_aggregate` | Mean of incoming messages per node |
| `EdgeConv` | Edge-conditioned message-passing layer: message = MLP(neighbor embedding ‖ edge feature ‖ direction) |
| `DiffGRN` | Input layer (215 → 64), two `EdgeConv` layers with residuals + LayerNorm, MLP decoder on [h_TF, h_target, h_TF ⊙ h_target, e] |

### 7. Training and evaluation
| Name | Purpose |
|---|---|
| `VARIANTS` | Model variants: `full`, `mlp_no_graph` (baseline: no message passing), and ablations `no_edge_feat`, `b1_only`, `t_feature`, `is_tf_only`, `no_is_tf`, `random_graph` |
| `early_precision_ratio` | Precision in the top-k predictions divided by the base rate |
| `Runner` | Holds one run's tensors and model; `train` (BCE, Adam, early stopping on validation AUPRC, p sampled each epoch), `evaluate` (AUROC, AUPRC, EPR), `scores`, `e` |

### 8. Differential score
| Name | Purpose |
|---|---|
| `all_positive_pairs` | All known edges across splits |
| `delta_scores` | Δ = σ(s(p_high)) − σ(s(p_low)) with the trained model |
| `differential` | Δ for all edges plus a permutation null (phenotype shuffled, Eq. 2 refit, same model); pooled p-values and BH q-values |
| `attach_lookup` | Coefficients for arbitrary edge lists |
| `bh` | Benjamini–Hochberg q-values |

### 9. Comparison methods for Δ
`rf_differential`: Optional GENIE3-style random-forest differential baseline (`do_rf=True`)
`compare_rankings`: Spearman correlation and top-k overlap between two edge rankings

### Runner, summaries, figures

 `run_all`: Full experiment loop (seeds × modes × conditions × variants); saves results after every run 
 `summarize`: Mean ± SD tables for link prediction and Δ

### SERGIO simulation
`setup_sergio`: Downloads and patches SERGIO, then imports it
`sample_subnetwork`: Random acyclic subnetwork of the leaf network (60 TFs; targets with ≥ 2 regulators)
`simulate_sergio`: Simulates expression for the real lines, with real phenotypes and genotypes; 10% of edges have strength K·exp(γP)
`detection_metrics`: AUROC / AUPRC of \|Δ\| and \|t(β₃)\| for recovering planted differential edges
`run_simulation`: Simulation → GNN training per seed and mode → detection metrics

## Code authorship

Developed by Maria: 
Project design: research question, use of mGRN+ as a prior graph, genotype × phenotype edge feature (Eqs. 2–3), differential score Δ (Eq. 1), choice of phenotypes, ablation and evaluation plan, SERGIO ground-truth idea
Data decisions: log-CPM, tail-mean contrast, sign alignment, separate traits, TF flag instead of motif features, report scope
Experiments and analysis: all runs, interpretation of results, report

Developed by Claude, Monitored by Maria:
Code in sections 0–12: drafted with an AI assistant (Claude, Anthropic); reviewed, commented, debugged and run on Kaggle by Maria Pugliese
Data acquisition: v4→v5 ID table (MaizeGDB), genotype data and sample key (from the Gemma Lab)

### Third-party code and data
- **SERGIO** (Dibaeinia & Sinha, 2020, *Cell Systems*; GPL-3.0) — downloaded at run time; not modified in this repository apart from numpy-compatibility patches applied when loading.
- **mGRN+** leaf network (Huo et al., 2025, *Molecular Plant*).
- **MaizeGDB** v4 → v5 gene model translation table.
- Expression, phenotype and genotype data: Gemma Lab, NC State University (not distributed)
