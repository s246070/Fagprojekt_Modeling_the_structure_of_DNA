````markdown
# dna_modelling

This project investigates the use of Latent Distance Models (LDMs) for modeling sparse single-cell ATAC-seq data. The data is represented as a bipartite graph between cells and chromatin features, where accessible regions are modeled as interactions between cells and peaks.

The goal of the project is to learn low-dimensional latent representations of cells and features, where distances in the latent space reflect the probability of chromatin accessibility. The model is evaluated using held-out accessibility links and compared with dimensionality-reduction approaches such as PCA and PeakVI.

The project focuses on predictive performance, biological interpretability of the learned latent space, and the effect of batching on scalability and model quality

## Project structure

The directory structure of the project looks like this:
```txt
├── data/                       # Data directory
│   ├── train_sets/             # premade training sets
│   ├── big_dataset
│   └── small_dataset
├── models/                     # Trained models
├── notebooks/                  # Jupyter notebooks
│   └── stat.ipynb              # Statistical evaluations
├── outfiles/                   # lsf outfiles
├── plots/
├── results/                    # Model performance results
├── src/                        # Source code
│   ├── benchmarking/           # 
│   │   ├── celltype.py
│   │   ├── Metrics.py
│   │   ├── plot_latentspace.py
│   │   └── plot_small_latentspace.py
│   ├── dna_modelling/          # LDM implementation
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── data.py         # Dataloader for all folders
│   │   │   ├── file_reader.py
│   │   │   ├── latent_space_visualization.py
│   │   │   └── visualize.py 
│   │   ├── __init__.py
│   │   ├── evaluate.py
│   │   ├── main_2.py           # full training and evaluation pipeline for premade training sets
│   │   ├── main.py             # full training and evaluation pipeline 
│   │   ├── model.py
│   │   └── train.py
│   ├── PeakVI/                 # PeakVI implementation
│   │   ├── __init__.py
│   │   ├── main_peakvi.py      # full training pipeline
│   │   ├── PeakVI.py
│   │   └── Reference_mapping.py
│   ├── scripts/
│   │   ├── templates/
│   │   │   ├── cpu_template.sh
│   │   │   └── gpu_template.sh
│   │   ├── bulk_jobs.sh
│   │   ├── bulk_peakvi_jobs.sh
│   │   ├── main_2.sh
│   │   ├── main.sh
│   │   ├── maketestsets.sh
│   │   └── PeakVi.sh
│   ├── simba/ 
│   └── __init__.py
├── .gitignore
├── .python-version
├── AGENTS.mf                   # Project commands, environment and usage documentation
├── pyproject.toml              # Python project file
├── README.md                   # Project README
└── tasks.py                    # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

````
