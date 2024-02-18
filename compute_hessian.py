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
from pyhessian import hessian


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


def _convert_to_hessian_dataloader(dataloader, cfg):
    assert (cfg.hessian_batch_size % cfg.mini_hessian_batch_size == 0)
    batch_num = cfg.hessian_batch_size // cfg.mini_hessian_batch_size

    if batch_num == 1:
        for data in dataloader:
            inputs, attention_masks, labels = data["input_ids"], data["attention_mask"], data["label"]
            inputs = torch.stack(inputs, dim=0).transpose(0, 1)
            attention_masks = torch.stack(attention_masks, dim=0).transpose(0, 1)
            hessian_dataloader = (inputs, attention_masks, labels)
            break
    else:
        hessian_dataloader = []
        for i, data in enumerate(dataloader):
            inputs, attention_masks, labels = data["input_ids"], data["attention_mask"], data["label"]
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


def _dump_to_json(cfg, top_eigenvalues, trace):
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    output_path = os.path.join(cfg.output_dir, "hessian.json")
    with open(output_path, "w") as f:
        json.dump({
            "top_eigenvalues": top_eigenvalues,
            "trace": trace
        }, f)


def _dump_numpy(cfg, top_eigenvectors, output_dir=None):
    output_dir = output_dir if output_dir else os.path.join(cfg.output_dir, "eigenvectors")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, eigenvector in enumerate(top_eigenvectors):
        if isinstance(eigenvector, np.ndarray):
            output_path = os.path.join(output_dir, f"{i}.npy")
            np.save(output_path, eigenvector)
        elif isinstance(eigenvector, list):
            _dump_numpy(cfg, eigenvector, output_dir=os.path.join(output_dir, str(i)))
        else:
            raise ValueError("Invalid type of eigenvector", type(eigenvector))


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
    hessian_dataloader, batch_num = _convert_to_hessian_dataloader(dataloader, cfg.dataloader)

    # model
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model, use_safetensors=True)
    model = ModelWrapper(model)
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=cfg.dataloader.device_ids)
    model.eval()

    # criterion
    criterion = torch.nn.CrossEntropyLoss()

    # hessian
    if batch_num == 1:
        hessian_comp = hessian(model,
                               criterion,
                               data=hessian_dataloader,
                               cuda=True)
    else:
        hessian_comp = hessian(model,
                               criterion,
                               dataloader=hessian_dataloader,
                               cuda=True)

    print('********** finish data loading and begin Hessian computation **********')
    top_eigenvalues, top_eigenvectors = hessian_comp.eigenvalues(top_n=cfg.hessian.top_n)
    top_eigenvectors = _convert_to_numpy(top_eigenvectors)
    trace = hessian_comp.trace()
    _dump_to_json(cfg, top_eigenvalues, trace)
    _dump_numpy(cfg, top_eigenvectors)
    print('********** finish Hessian computation **********')


if __name__ == "__main__":
    main()
