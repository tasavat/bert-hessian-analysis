"""
This code is used for finetuning the BERT model.
Training configurations can be found under config/ directory.
参考: https://huggingface.co/docs/transformers/training
"""

import os

import evaluate
import hydra
import numpy as np
import torch.optim as optim
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.integrations import MLflowCallback


def _tokenize_function(example, tokenizer=None, text_column_name=None):
    return tokenizer(example[text_column_name], padding="max_length", truncation=True)


def _compute_metrics(metric, eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


@hydra.main(config_path="config/finetune", config_name="")
def main(cfg):
    # load dataset
    dataset = load_dataset(cfg.dataset.name, cfg.dataset.task_name)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)
    tokenized_dataset = dataset.map(
        _tokenize_function,
        fn_kwargs={
            "tokenizer": tokenizer,
            "text_column_name": cfg.dataset.text_column_name,
        },
        batched=True,
    )

    # load model
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.prefix + cfg.model.name, num_labels=cfg.dataset.num_labels
    )

    # define training arguments
    training_args = TrainingArguments(
        output_dir=cfg.training_args.output_dir,
        evaluation_strategy=cfg.training_args.evaluation_strategy,
        per_device_train_batch_size=cfg.training_args.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training_args.per_device_eval_batch_size,
        max_grad_norm=cfg.training_args.max_grad_norm,
        num_train_epochs=cfg.training_args.num_train_epochs,
        save_strategy=cfg.training_args.save_strategy,
    )

    # optimizer and scheduler
    optimizer = getattr(optim, cfg.optim.name)
    optimizer = optimizer(model.parameters(), **cfg.optim.params)
    scheduler = getattr(optim.lr_scheduler, cfg.scheduler.name)
    scheduler = scheduler(optimizer)

    # callbacks
    os.environ["MLFLOW_EXPERIMENT_NAME"] = cfg.mlflow.experiment_name
    callbacks = [MLflowCallback()]

    # define metric
    metric = evaluate.load("accuracy")
    compute_metrics = lambda eval_pred: _compute_metrics(metric, eval_pred)

    # define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        optimizers=(optimizer, scheduler),
        callbacks=callbacks,
        compute_metrics=compute_metrics,
    )

    # train model
    trainer.train()


if __name__ == "__main__":
    main()
