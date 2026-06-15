Guidance for autonomous coding agents
Read this before writing, editing, or executing anything in this repo.

# Relevant commands

* The project uses `uv` for management of virtual environments and dependencies.
* The project uses separate virtual environments for CPU, GPU, and SIMBA usage:
  * `.venv-cpu` is used for CPU-only work.
  * `.venv-gpu` is used for CUDA/GPU work.
  * `.venv-simba` is used for SIMBA-specific work.
* The project uses one shared `uv.lock` file. Do not create separate lock files for CPU, GPU, or SIMBA.
* Do not install both CPU and GPU extras into the same environment.
* Do not run `uv sync --all-extras`, because the `cpu` and `gpu` extras are mutually exclusive.
* For named environments, prefer `UV_PROJECT_ENVIRONMENT=<env-name> uv sync ...` when possible, because it avoids ambiguity about which environment `uv` is syncing into.

## Environment installation

### Clean old environments

If starting from scratch or after changing PyTorch-related dependencies, remove old environments:

```bash
rm -rf .venv .venv-cpu .venv-gpu .venv-simba
```

Removing `uv.lock` is usually not necessary. Only remove it if the dependency structure has changed significantly and the lock file needs to be regenerated:

```bash
rm -f uv.lock
```

### Create the CPU environment

Use this environment for preprocessing, Scanpy work, test-set generation, CPU training, and CPU-only jobs:

```bash
UV_PROJECT_ENVIRONMENT=.venv-cpu uv sync --extra cpu --python 3.12
```

Alternative activate/deactivate workflow:

```bash
uv venv .venv-cpu --python 3.12
source .venv-cpu/bin/activate
uv sync --extra cpu --active --python 3.12
deactivate
```

To use the CPU environment later:

```bash
source .venv-cpu/bin/activate
```

### Create the GPU environment

Use this environment for CUDA/GPU training jobs:

```bash
UV_PROJECT_ENVIRONMENT=.venv-gpu uv sync --extra gpu --python 3.12
```

Alternative activate/deactivate workflow:

```bash
uv venv .venv-gpu --python 3.12
source .venv-gpu/bin/activate
uv sync --extra gpu --active --python 3.12
deactivate
```

To use the GPU environment later:

```bash
source .venv-gpu/bin/activate
```

### Create the SIMBA environment

Use this environment explicitly for SIMBA. SIMBA should be kept separate from the normal CPU and GPU environments because its dependencies may conflict with the newer Scanpy/scVI/PyTorch stack.

Preferred command:

```bash
UV_PROJECT_ENVIRONMENT=.venv-simba uv sync --extra cpu --extra simba --python 3.10
```

Alternative activate/deactivate workflow:

```bash
uv venv .venv-simba --python 3.10
source .venv-simba/bin/activate
UV_PROJECT_ENVIRONMENT=.venv-simba uv sync --extra cpu --extra simba --python 3.10
deactivate
```

Avoid using plain `uv sync --extra cpu --extra simba` for the SIMBA environment, because it may sync into the default project environment instead of `.venv-simba`.

To use the SIMBA environment later:

```bash
source .venv-simba/bin/activate
```

### Verify PyTorch installation

For CPU:

```bash
source .venv-cpu/bin/activate
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
deactivate
```

Expected CUDA availability:

```text
False
```

For GPU:

```bash
source .venv-gpu/bin/activate
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
deactivate
```

Expected CUDA availability on a GPU node:

```text
True
```

### Verify SIMBA installation

For SIMBA:

```bash
source .venv-simba/bin/activate
python --version
python -c "import simba; print('SIMBA imported successfully')"
deactivate
```

Expected Python version:

```text
Python 3.10.x
```

## Running commands

* To install regular project packages, prefer editing `pyproject.toml` and then running the relevant sync command:
  * CPU: `UV_PROJECT_ENVIRONMENT=.venv-cpu uv sync --extra cpu --python 3.12`
  * GPU: `UV_PROJECT_ENVIRONMENT=.venv-gpu uv sync --extra gpu --python 3.12`
  * SIMBA: `UV_PROJECT_ENVIRONMENT=.venv-simba uv sync --extra cpu --extra simba --python 3.10`
* To add a new shared dependency, use:

```bash
uv add <package-name>
```

Then sync the relevant environment again with the relevant extra.

* To run Python scripts, activate the correct environment first, then use:

```bash
python <script-name>.py
```

or use `uv run` with the relevant extra:

```bash
uv run --extra cpu <script-name>.py
```

for CPU, and:

```bash
uv run --extra gpu <script-name>.py
```

for GPU.

* To run other commands related to Python, activate the correct environment first and run the command normally, or prefix with `uv run --extra cpu` / `uv run --extra gpu`.

## Testing

* The project uses `pytest` for testing. To run tests in the CPU environment:

```bash
source .venv-cpu/bin/activate
uv run --extra cpu pytest tests/
deactivate
```

GPU-specific tests should be run in the GPU environment:

```bash
source .venv-gpu/bin/activate
uv run --extra gpu pytest tests/
deactivate
```

## Linting and formatting

* The project uses `ruff` for linting and formatting.
* To format code:

```bash
uv run ruff format .
```

* To lint code and apply safe fixes:

```bash
uv run ruff check . --fix
```

## Task management

* The project uses `invoke` for task management.
* To see available tasks, use:

```bash
uv run invoke --list
```

or refer to the `tasks.py` file.

# HPC usage

* CPU jobs should activate `.venv-cpu`.
* GPU jobs should activate `.venv-gpu`.
* SIMBA jobs should activate `.venv-simba`.
* Do not use the CUDA/GPU PyTorch environment on CPU-only HPC jobs, because it can fail with missing CUDA/cuDNN libraries such as `libcudnn.so.9`.
* Do not use the CPU PyTorch environment for GPU training jobs, because `torch.cuda.is_available()` will be `False`.
* SIMBA is expected to be CPU-oriented, so prefer CPU nodes for SIMBA runs unless there is a specific reason to use a GPU node.

Example CPU job setup:

```bash
module purge
module load python3/3.12.11

source .venv-cpu/bin/activate

python src/dna_modelling/main.py
```

Example GPU job setup:

```bash
module purge
module load python3/3.12.11

source .venv-gpu/bin/activate

python src/dna_modelling/main.py
```

Example SIMBA job setup:

```bash
module purge
module load python3/3.10.20

source .venv-simba/bin/activate

python src/dna_modelling/run_simba.py
```

# Code style

* Follow existing code style.
* Keep line length within 120 characters.
* Use f-strings for formatting.
* Use type hints.
* Do not add inline comments unless absolutely necessary.

# Documentation

* If the project has a `docs/` folder, update documentation there as needed.
* In this case the project uses `mkdocs` for documentation. To build the docs locally, use:

```bash
uv run mkdocs serve
```

* Use existing docstring style.
* Ensure all functions and classes have docstrings.
* Use Google style for docstrings.
* Update this `AGENTS.md` file if any new tools or commands are added to the project.
