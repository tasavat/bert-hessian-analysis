"""
This code plots an optimization trajectory on a 2D plot of the objective function.

References:
https://arxiv.org/pdf/1908.05620.pdf
"""

from __future__ import annotations

import itertools
import os
import warnings
from copy import deepcopy

import hydra
from tqdm import tqdm

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import torch

torch.manual_seed(0)

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        self.embeddings = self.model.bert.embeddings
        self.encoder = self.model.bert.encoder

    def forward(self, inputs, attention_mask):
        outputs = self.model(inputs, attention_mask=attention_mask)
        logits = outputs.logits
        return logits


def _tokenize_function(example, tokenizer=None, text_column_name=None):
    return tokenizer(example[text_column_name], padding="max_length", truncation=True)


def _extract_parameters(model_name, use_safetensors=False):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, use_safetensors=use_safetensors
    )
    params = list(model.parameters())
    return params


def _get_params(model_orig, model_perb, direction, alpha):
    model_orig, model_perb = model_orig.cpu(), model_perb.cpu()
    for m_orig, m_perb, d in zip(
        model_orig.parameters(), model_perb.parameters(), direction
    ):
        m_perb.data = m_orig.data + alpha * d
    model_orig, model_perb = model_orig.cuda(), model_perb.cuda()
    return model_perb


def _plot_loss_landscape(
    losses, perterbations, output_dir, suffix="", set_lim=False, z_lim=5
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    X, Y = np.meshgrid(perterbations, perterbations)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, losses, cmap="viridis", alpha=0.8)
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


def _cal_cossim(v1: list[torch.Tensor], v2: list[torch.Tensor]) -> float:
    v1 = torch.cat([v.view(-1) for v in v1]).reshape(1, -1)
    v2 = torch.cat([v.view(-1) for v in v2]).reshape(1, -1)
    return torch.nn.functional.cosine_similarity(v1, v2).item()


def _cal_magnitude(v1: list[torch.Tensor]) -> float:
    v = torch.cat([v.view(-1) for v in v1])
    return torch.norm(v).item()


def _cal_projection_coeff(v: list[torch.Tensor], e: list[torch.Tensor]) -> tuple:
    v_cos = _cal_cossim(v, e)
    v_mag = _cal_magnitude(v)
    e_mag = _cal_magnitude(e)
    d_alpha = v_cos * v_mag / e_mag
    d_beta = np.sqrt((v_mag / e_mag) ** 2 - d_alpha**2)
    return d_alpha, d_beta


def _plot_loss_landscape_with_trajectory(
    losses,
    perterbations,
    trajectory_losses,
    trajectory_perterbations,
    output_dir,
    suffix="",
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    X, Y = np.meshgrid(perterbations, perterbations)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, losses, cmap="viridis", alpha=0.8)
    ax.set_xlabel("Perturbation 1")
    ax.set_ylabel("Perturbation 2")
    ax.set_zlabel("Loss")

    trajectory_perterbations = np.array(trajectory_perterbations)
    ax.plot(
        trajectory_perterbations[:, 1],
        trajectory_perterbations[:, 0],
        trajectory_losses,
        marker="o",
        color="r",
    )

    plt.savefig(
        os.path.join(output_dir, f"loss_landscape_with_trajectory_{suffix}.png")
    )
    ax.view_init(30, 45)
    plt.savefig(
        os.path.join(output_dir, f"loss_landscape_with_trajectory_{suffix}_2.png")
    )
    ax.view_init(90, -90)
    plt.savefig(
        os.path.join(output_dir, f"loss_landscape_with_trajectory_{suffix}_3.png")
    )
    plt.close()
    print(f"Loss landscape with trajectory plot saved to {output_dir}.")


@hydra.main(config_path="config/plot/trajectory", config_name="")
def main(cfg):
    # Dataset
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

    # Dataloader
    dataloader = torch.utils.data.DataLoader(
        tokenized_dataset["train"],
        batch_size=cfg.dataloader.mini_batch_size,
        shuffle=True,
    )
    assert (
        cfg.dataloader.batch_size % cfg.dataloader.mini_batch_size == 0
    ), "Batch size must be divisible by mini batch size."

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.models[0], use_safetensors=cfg.use_safetensors[0]
    )
    model = ModelWrapper(model)
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=cfg.dataloader.device_ids)
    model.eval()

    # Extract parameters
    params_0 = _extract_parameters(
        cfg.models[0], use_safetensors=cfg.use_safetensors[0]
    )
    params_1 = _extract_parameters(
        cfg.models[1], use_safetensors=cfg.use_safetensors[1]
    )
    params_2 = _extract_parameters(
        cfg.models[2], use_safetensors=cfg.use_safetensors[2]
    )
    params_diff_0_1 = [p1 - p0 for p0, p1 in zip(params_0, params_1)]
    params_diff_0_2 = [p1 - p0 for p0, p1 in zip(params_0, params_2)]
    scale_0_2 = _cal_magnitude(params_diff_0_1) / _cal_magnitude(params_diff_0_2)
    params_diff_0_2 = [p * scale_0_2 for p in params_diff_0_2]
    print(
        f"Cosine similarity between two axes: {_cal_cossim(params_diff_0_1, params_diff_0_2)}"
    )

    model_orig = deepcopy(model)
    for perterbations in cfg.perterbation_sets:
        # Plot 2D Landscape
        print("Computing loss landscape...")
        losses = np.zeros((len(perterbations), len(perterbations)))
        for p1, p2 in tqdm(itertools.product(perterbations, perterbations)):
            model = deepcopy(model_orig)
            model = _get_params(model, model, params_diff_0_1, p1)
            model = _get_params(model, model, params_diff_0_2, p2)

            for batch_idx, batch in enumerate(dataloader):
                if (
                    batch_idx
                    >= cfg.dataloader.batch_size // cfg.dataloader.mini_batch_size
                ):
                    break
                inputs = torch.stack(batch["input_ids"]).transpose(0, 1).cuda()
                attention_mask = (
                    torch.stack(batch["attention_mask"]).transpose(0, 1).cuda()
                )
                labels = batch["label"].cuda()
                with torch.no_grad():
                    logits = model(inputs, attention_mask)
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                    loss = loss.cpu().item()
                losses[perterbations.index(p1), perterbations.index(p2)] += loss
            losses[perterbations.index(p1), perterbations.index(p2)] /= batch_idx

        _plot_loss_landscape(
            losses,
            perterbations,
            cfg.output_dir,
            suffix=f"{perterbations[0]}_{perterbations[-1]}",
        )
        _plot_loss_landscape(
            losses,
            perterbations,
            cfg.output_dir,
            suffix=f"{perterbations[0]}_{perterbations[-1]}",
            set_lim=True,
        )

        # save loss landscape
        np.save(
            os.path.join(
                cfg.output_dir,
                f"loss_landscape_{perterbations[0]}_{perterbations[-1]}.npy",
            ),
            losses,
        )

        # load loss landscape
        losses = np.load(
            os.path.join(
                cfg.output_dir,
                f"loss_landscape_{perterbations[0]}_{perterbations[-1]}.npy",
            )
        )

        # Plot Optimization Trajectory
        print("Computing optimization trajectory...")
        trajectory_losses, trajectory_perterbations = [], []
        for trajectory_model in tqdm(cfg.trajectory_models):
            trajectory_params = _extract_parameters(
                trajectory_model, use_safetensors=True
            )
            trajectory_params = [p1 - p0 for p0, p1 in zip(params_0, trajectory_params)]
            trajectory_perterbation = _cal_projection_coeff(
                trajectory_params, params_diff_0_1
            )
            trajectory_perterbations.append(trajectory_perterbation)

            model = deepcopy(model_orig)
            model = _get_params(
                model, model, params_diff_0_1, trajectory_perterbation[0]
            )
            model = _get_params(
                model, model, params_diff_0_2, trajectory_perterbation[1]
            )

            trajectory_loss = []
            for batch_idx, batch in enumerate(dataloader):
                if (
                    batch_idx
                    >= cfg.dataloader.batch_size // cfg.dataloader.mini_batch_size
                ):
                    break
                inputs = torch.stack(batch["input_ids"]).transpose(0, 1).cuda()
                attention_mask = (
                    torch.stack(batch["attention_mask"]).transpose(0, 1).cuda()
                )
                labels = batch["label"].cuda()
                with torch.no_grad():
                    logits = model(inputs, attention_mask)
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                    loss = loss.cpu().item()
                trajectory_loss.append(loss)
            trajectory_losses.append(np.mean(trajectory_loss))

        _plot_loss_landscape_with_trajectory(
            losses,
            perterbations,
            trajectory_losses,
            trajectory_perterbations,
            cfg.output_dir,
            suffix=f"{perterbations[0]}_{perterbations[-1]}",
        )


if __name__ == "__main__":
    main()
