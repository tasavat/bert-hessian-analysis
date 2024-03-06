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
    python finetune_model.py --config-path {CONFIG_PATH} --config-name {CONFIG_NAME}

    # For example
    python finetune_model.py --config-path config/finetune --config-name bert_adam
    ```
  - Minitor training results with MLFlow
    ```
    mlflow ui --port {PORT}
    ```

- Compute hessian
  - Replace `CONFIG_NAME` with config name under `config/hessian/`
    ```
    python compute_hessian.py --config-path {CONFIG_PATH} --config-name {CONFIG_NAME}

    # For example
    python compute_hessian.py --config-path config/hessian/params --config-name bert_pretrained
    python compute_hessian.py --config-path config/hessian/inputs --config-name bert_pretrained
    ```
  - Results are under `output/hessian/` (by default)

- Plot eigenvalues and loss landscape
  - Replace `CONFIG_NAME` with config name under `config/plot/`
    ```
    python plot.py --config-path {CONFIG_PATH} --config-name {CONFIG_NAME}

    # For example
    python plot_params.py --config-path config/plot/params --config-name bert_pretrained
    python plot_inputs.py --config-path config/plot/inputs --config-name bert_pretrained
    ```
  - Results are under `output/plot/` (by default)

---