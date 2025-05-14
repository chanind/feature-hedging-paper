# Feature Hedging Paper

Code for the paper "Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders".

## Repo structure

This Repo contains the experiments run in the paper in `/experiments`, with toy model experiments in the `/notebooks` dir. Likely you don't want to directly run the experiments we did verbatim as its expensive to train so many SAEs, but if you do, the experiments with all hyperparams we used are there for reference.

Potentially more useful are the matryoshka SAE implementations in the `hedging_paper/saes` dir and the evaluations in the `hedging_paper/evals` dir. For running your own toy model experiments, see `notebooks/single_latent_hedging_plots.ipynb`.

## Setup

This project uses Poetry for dependency management. To install the dependencies, run:

```bash
poetry install
```

### Tests

To run the tests, run:

```bash
poetry run pytest
```

### Linting / Formatting

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting. To set this up with VSCode, install the ruff plugina and add the following to `.vscode/settings.json`:

```json
{
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll": "explicit",
      "source.organizeImports": "explicit"
    },
    "editor.defaultFormatter": "charliermarsh.ruff"
  },
  "notebook.formatOnSave.enabled": true
}
```

### Pre-commit hook

There's a pre-commit hook that will run ruff and pyright on each commit. To install it, run:

```bash
poetry run pre-commit install
```

### Poetry tips

Below are some helpful tips for working with Poetry:

- Install a new main dependency: `poetry add <package>`
- Install a new development dependency: `poetry add --dev <package>`
  - Development dependencies are not required for the main code to run, but are for things like linting/type-checking/etc...
- Update the lockfile: `poetry lock`
- Run a command using the virtual environment: `poetry run <command>`
- Run a Python file from the CLI as a script (module-style): `poetry run python -m hedging_paper.path.to.file`
