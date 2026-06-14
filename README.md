````markdown
# dna_modelling

This project investigates the use of Latent Distance Models (LDMs) for modeling sparse single-cell ATAC-seq data. The data is represented as a bipartite graph between cells and chromatin features, where accessible regions are modeled as interactions between cells and peaks.

The goal of the project is to learn low-dimensional latent representations of cells and features, where distances in the latent space reflect the probability of chromatin accessibility. The model is evaluated using held-out accessibility links and compared with dimensionality-reduction approaches such as PCA and PeakVI.

The project focuses on predictive performance, biological interpretability of the learned latent space, and the effect of batching on scalability and model quality

## Project structure

The directory structure of the project looks like this:
```txt
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── src/                      # Source code
│   ├── dna_modelling/        # LDM implementation
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
│   ├── PeakVI/               # PeakVI implementation
│   │   ├── __init__.py
│   │   ├── main_peakvi.py
│   │   ├── 
│   │   ├── 
│   │   ├── 
│   │   └──
│   └── Simba/ 
├── .gitignore
├── pyproject.toml            # Python project file
├── README.md                 # Project README
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

````
