"""
参考: https://github.com/amirgholami/PyHessian/blob/master/example_pyhessian_analysis.py
"""

import hydra
import os
import json

import numpy as np
import torch

torch.manual_seed(0)

from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from pyhessian import hessian


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


def _convert_to_hessian_dataloader(dataloader, backprop_inputs, cfg):
    assert cfg.hessian_batch_size % cfg.mini_hessian_batch_size == 0
    batch_num = cfg.hessian_batch_size // cfg.mini_hessian_batch_size

    if batch_num == 1:
        for data in dataloader:
            inputs, attention_masks, labels = (
                data["input_ids"],
                data["attention_mask"],
                data["label"],
            )
            inputs = torch.stack(inputs, dim=0).transpose(0, 1)
            attention_masks = torch.stack(attention_masks, dim=0).transpose(0, 1)
            hessian_dataloader = (inputs, attention_masks, labels)
            break
    elif backprop_inputs:
        # specify the index of the data to be used for Hessian computation
        for index, data in enumerate(dataloader):
            batch_num = 1
            if index != cfg.data_index:
                continue
            inputs, attention_masks, labels = (
                data["input_ids"],
                data["attention_mask"],
                data["label"],
            )
            inputs = torch.stack(inputs, dim=0).transpose(0, 1)
            attention_masks = torch.stack(attention_masks, dim=0).transpose(0, 1)
            hessian_dataloader = (inputs, attention_masks, labels)
            break
    else:
        hessian_dataloader = []
        for i, data in enumerate(dataloader):
            inputs, attention_masks, labels = (
                data["input_ids"],
                data["attention_mask"],
                data["label"],
            )
            inputs = torch.stack(inputs, dim=0).transpose(0, 1)
            attention_masks = torch.stack(attention_masks, dim=0).transpose(0, 1)
            hessian_dataloader.append((inputs, attention_masks, labels))
            if i == batch_num - 1:
                break
    return hessian_dataloader, batch_num


def _convert_to_numpy(tensor_list):
    for i, tensor in enumerate(tensor_list):
        if isinstance(tensor, list):
            tensor_list[i] = _convert_to_numpy(tensor)
        else:
            tensor_list[i] = tensor.cpu().detach().numpy()
    return tensor_list


def _dump_hessian_json(cfg, top_eigenvalues, trace):
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    output_path = os.path.join(cfg.output_dir, "hessian.json")
    with open(output_path, "w") as f:
        json.dump({"top_eigenvalues": top_eigenvalues, "trace": trace}, f)


def _dump_eigenvectors(cfg, top_eigenvectors, output_dir=None):
    output_dir = (
        output_dir if output_dir else os.path.join(cfg.output_dir, "eigenvectors")
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, eigenvector in enumerate(top_eigenvectors):
        if isinstance(eigenvector, np.ndarray):
            output_path = os.path.join(output_dir, f"{i}.npy")
            np.save(output_path, eigenvector)
        elif isinstance(eigenvector, list):
            _dump_eigenvectors(
                cfg, eigenvector, output_dir=os.path.join(output_dir, str(i))
            )
        else:
            raise ValueError("Invalid type of eigenvector", type(eigenvector))


def _dump_data(cfg, embedding, attention_masks, targets):
    _dump_embedding(cfg, embedding)
    _dump_attention_masks(cfg, attention_masks)
    _dump_targets(cfg, targets)


def _dump_embedding(cfg, embedding):
    output_path = os.path.join(cfg.output_dir, "embedding.npy")
    if type(embedding) is torch.Tensor:
        embedding = embedding.cpu().detach().numpy()
    np.save(output_path, embedding)


def _dump_attention_masks(cfg, attention_masks):
    output_path = os.path.join(cfg.output_dir, "attention_masks.npy")
    if type(attention_masks) is torch.Tensor:
        attention_masks = attention_masks.cpu().detach().numpy()
    np.save(output_path, attention_masks)


def _dump_targets(cfg, targets):
    output_path = os.path.join(cfg.output_dir, "targets.npy")
    if type(targets) is torch.Tensor:
        targets = targets.cpu().detach().numpy()
    np.save(output_path, targets)


@hydra.main(config_path="config/hessian", config_name="")
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
        batch_size=cfg.dataloader.mini_hessian_batch_size,
        shuffle=True,
    )
    hessian_dataloader, batch_num = _convert_to_hessian_dataloader(
        dataloader, cfg.hessian.backprop_inputs, cfg.dataloader
    )

    # model
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model, use_safetensors=cfg.use_safetensors
    )
    model = ModelWrapper(model)
    model = model.cuda()
    if not cfg.hessian.backprop_inputs:
        model = torch.nn.DataParallel(model, device_ids=cfg.dataloader.device_ids)
    model.eval()

    # criterion
    criterion = torch.nn.CrossEntropyLoss()

    # hessian
    if batch_num == 1:
        hessian_comp = hessian(
            model,
            criterion,
            data=hessian_dataloader,
            cuda=True,
            backprop_inputs=cfg.hessian.backprop_inputs,
        )
    else:
        hessian_comp = hessian(
            model,
            criterion,
            dataloader=hessian_dataloader,
            cuda=True,
            backprop_inputs=cfg.hessian.backprop_inputs,
        )

    print("********** finish data loading and begin Hessian computation **********")
    top_eigenvalues, top_eigenvectors = hessian_comp.eigenvalues(
        top_n=cfg.hessian.top_n
    )
    top_eigenvectors = _convert_to_numpy(top_eigenvectors)
    trace = hessian_comp.trace()
    _dump_hessian_json(cfg, top_eigenvalues, trace)
    _dump_eigenvectors(cfg, top_eigenvectors)
    if cfg.hessian.backprop_inputs:
        _dump_data(
            cfg,
            hessian_comp.embeddings,
            hessian_comp.attention_masks,
            hessian_comp.targets,
        )
    print("********** finish Hessian computation **********")


if __name__ == "__main__":
    main()
