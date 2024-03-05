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
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        self.embeddings = self.model.bert.embeddings
        self.encoder = self.model.bert.encoder

    def forward(
        self,
        inputs,
        attention_mask,
        embed_forward=False,
        encode_forward=False,
        embeddings=None,
    ):
        if embed_forward:
            embeddings = self._embed_forward(inputs)
            return embeddings
        if encode_forward:
            assert embeddings is not None, "embeddings must be provided"
            return self._encode_forward(inputs, embeddings, attention_mask)
        outputs = self.model(inputs, attention_mask=attention_mask)
        logits = outputs.logits
        return logits

    def _embed_forward(self, inputs):
        with torch.no_grad():
            embeddings = self.embeddings(
                input_ids=inputs,
                position_ids=None,
                token_type_ids=None,
                inputs_embeds=None,
                past_key_values_length=0,
            )
        return embeddings

    def _encode_forward(self, inputs, embeddings, attention_mask):
        input_shape = inputs.size()
        extended_attention_mask = self.model.get_extended_attention_mask(
            attention_mask, input_shape
        )
        head_mask = self.model.get_head_mask(None, self.model.config.num_hidden_layers)
        output_attentions = self.model.config.output_attentions
        output_hidden_states = self.model.config.output_hidden_states
        return_dict = self.model.config.use_return_dict

        encoder_outputs = self.encoder(
            embeddings,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.model.bert.pooler(sequence_output)
            if self.model.bert.pooler is not None
            else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        bert_outputs = BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
        pooled_output = bert_outputs[1]
        pooled_output = self.model.dropout(pooled_output)
        logits = self.model.classifier(pooled_output)
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
    filenames = [f for f in os.listdir(filedir) if f.endswith(".npy")]
    filenames.sort(key=lambda x: int(x.split(".")[0]))

    eigenvectors = [np.load(os.path.join(filedir, f)) for f in filenames]
    return eigenvectors


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
    assert (
        cfg.dataloader.batch_size % cfg.dataloader.mini_batch_size == 0
    ), "Batch size must be divisible by mini batch size."

    # model
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model, use_safetensors=cfg.use_safetensors
    )
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
                attention_mask = (
                    torch.stack(batch["attention_mask"]).transpose(0, 1).cuda()
                )
                labels = batch["label"].cuda()
                with torch.no_grad():
                    logits = model(inputs, attention_mask)
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                    loss = loss.cpu().item()
                losses[perterbations.index(p1), perterbations.index(p2)] += loss
        for i, j in itertools.product(
            range(len(perterbations)), range(len(perterbations))
        ):
            losses[i, j] /= batch_idx
        # plot loss landscape
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


if __name__ == "__main__":
    main()
