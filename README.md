## Installation

```
# install conda environment
conda env create -f environment.yml

# activate
conda activate bert-hessian-analysis

# replace hessian.py
cp pyhessian/hessian.py {ANACONDA_LIB_DIR}/pyhessian/hessian.py
```

## Reproduce Steps

- Finetuning BERT models
  - Replace `CONFIG_NAME` with config name under `config/finetune/`
    ```
    python finetune_model.py --config-name {CONFIG_NAME}

    # For example
    python finetune_model.py --config-name bert_adam
    ```
  - Minitor training results with MLFlow
    ```
    mlflow ui --port {PORT}
    ```

- Compute hessian
  - Replace `CONFIG_NAME` with config name under `config/hessian/`
    ```
    python compute_hessian.py --config-name {CONFIG_NAME}

    # For example
    python compute_hessian.py --config-name bert_pretrained
    ```
  - Results are under `output/hessian/` (by default)

- Plot eigenvalues and loss landscape
  - Replace `CONFIG_NAME` with config name under `config/plot/`
    ```
    python plot.py --config-name {CONFIG_NAME}

    # For example
    python plot.py --config-name bert_pretrained
    ```
  - Results are under `output/plot/` (by default)

---