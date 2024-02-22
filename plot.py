from __future__ import annotations
from copy import deepcopy
from tqdm import tqdm

import hydra
import os
import json
import itertools

import numpy as np
import matplotlib.pyplot as plt

import torch
torch.manual_seed(0)

from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs, attention_mask):
        outputs = self.model(inputs, attention_mask=attention_mask)
        logits = outputs.logits
        return logits


def _tokenize_function(example, tokenizer=None, text_column_name=None):
    return tokenizer(example[text_column_name], padding="max_length", truncation=True)


def _plot_eigenvalues(eigenvalues, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.plot([np.abs(e) for e in eigenvalues])
    plt.title("Top eigenvalues of the Hessian (absolute value)")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.savefig(os.path.join(output_dir, "eigenvalues.png"))
    plt.close()
    print(f"Top Eigenvalues plot saved to {output_dir}.")


def _load_eigenvector(filedir: str, index: int) -> list[np.ndarray]:
    """
    Downloads the eigenvectors from the file directory and returns them as a list of numpy arrays.
    """
    filedir = os.path.join(filedir, str(index))
    assert os.path.exists(filedir), f"File directory {filedir} does not exist."
    filenames = [f for f in os.listdir(filedir) if f.endswith('.npy')]
    filenames.sort(key=lambda x: int(x.split('.')[0]))
    
    eigenvectors = [np.load(os.path.join(filedir, f)) for f in filenames]
    return eigenvectors


def _get_params(model_orig,  model_perb, direction, alpha):
    model_orig, model_perb = model_orig.cpu(), model_perb.cpu()
    for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), direction):
        m_perb.data = m_orig.data + alpha * d
    model_orig, model_perb = model_orig.cuda(), model_perb.cuda()
    return model_perb


def _plot_loss_landscape(
    losses, 
    perterbations, 
    output_dir, 
    suffix="", 
    set_lim=False,
    z_lim=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    X, Y = np.meshgrid(perterbations, perterbations)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, losses, cmap='viridis', alpha=0.8)
    ax.set_xlabel("Perturbation 1")
    ax.set_ylabel("Perturbation 2")
    ax.set_zlabel("Loss")
    if set_lim:
        ax.set_zlim(0, z_lim)
        suffix += "_z-lim"
    plt.savefig(os.path.join(output_dir, f"loss_landscape_{suffix}.png"))
    ax.view_init(30, 45)
    plt.savefig(os.path.join(output_dir, f"loss_landscape_{suffix}_2.png"))
    plt.close()
    print(f"Loss landscape plot saved to {output_dir}.") 


@hydra.main(config_path="config/plot", config_name="")
def main(cfg):
    # dataset
    dataset = load_dataset(cfg.dataset.name, cfg.dataset.task_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)
    tokenized_dataset = dataset.map(
        _tokenize_function,
        fn_kwargs={
            "tokenizer": tokenizer,
            "text_column_name": cfg.dataset.text_column_name,
        },
        batched=True,
    )
    
    # dataloader
    dataloader = torch.utils.data.DataLoader(
        tokenized_dataset["train"], 
        batch_size=cfg.dataloader.mini_batch_size,
        shuffle=True,
    )
    assert cfg.dataloader.batch_size % cfg.dataloader.mini_batch_size == 0, "Batch size must be divisible by mini batch size."

    # model
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model, use_safetensors=True)
    model = ModelWrapper(model)
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=cfg.dataloader.device_ids)
    model.eval()

    # plot hessian eigenvalues
    with open(os.path.join(cfg.hessian_dir, "hessian.json"), "r") as f:
        hessian = json.load(f)
        top_eigenvalues = hessian["top_eigenvalues"]
    _plot_eigenvalues(top_eigenvalues, cfg.output_dir)

    # perturb model and compute loss
    model_orig = deepcopy(model)
    eigenvector_dir = os.path.join(cfg.hessian_dir, "eigenvectors")
    eigenvector_1 = _load_eigenvector(eigenvector_dir, 0)
    eigenvector_2 = _load_eigenvector(eigenvector_dir, 1)
    
    perterbation_sets = cfg.perterbation_sets
    for perterbations in perterbation_sets:
        losses = np.zeros((len(perterbations), len(perterbations)))
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= cfg.dataloader.batch_size // cfg.dataloader.mini_batch_size:
                break
            for p1, p2 in tqdm(itertools.product(perterbations, perterbations)):
                model = deepcopy(model_orig)
                model = _get_params(model, model, eigenvector_1, p1)
                model = _get_params(model, model, eigenvector_2, p2)
                inputs = torch.stack(batch["input_ids"]).transpose(0, 1).cuda()
                attention_mask = torch.stack(batch["attention_mask"]).transpose(0, 1).cuda()
                labels = batch["label"].cuda()
                with torch.no_grad():
                    logits = model(inputs, attention_mask)
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                    loss = loss.cpu().item()
                losses[perterbations.index(p1), perterbations.index(p2)] += loss
        for i, j in itertools.product(range(len(perterbations)), range(len(perterbations))):
            losses[i, j] /= batch_idx
        # plot loss landscape
        _plot_loss_landscape(losses, perterbations, cfg.output_dir, suffix=f"{perterbations[0]}_{perterbations[-1]}")
        _plot_loss_landscape(losses, perterbations, cfg.output_dir, suffix=f"{perterbations[0]}_{perterbations[-1]}", set_lim=True)
    

if __name__ == "__main__":
    main()
