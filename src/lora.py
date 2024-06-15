import argparse
import dataclasses
import sys

import datasets
import numpy as np

from datasets import load_dataset
from peft import AutoPeftModelForSequenceClassification, TaskType, PeftModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerBase, PreTrainedModel
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments


@dataclasses.dataclass
class Dataset:
    tokenizer: PreTrainedTokenizerBase
    tokenized_dataset: dict[str, datasets.arrow_dataset.Dataset]
    num_labels: int
    id2label: dict[int, str]
    label2id: dict[str, int]


def get_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_sms_dataset(model_name: str) -> Dataset:
    tokenizer = get_tokenizer(model_name=model_name)

    # The sms_spam dataset only has a train split
    dataset = load_dataset("sms_spam", split="train").train_test_split(
        test_size=0.2, shuffle=True, seed=23)

    splits = ["train", "test"]

    tokenized_dataset = {}
    for split in splits:
        tokenized_dataset[split] = dataset[split].map(lambda x: tokenizer(x["sms"], truncation=True), batched=True)

    dataset = Dataset(
        tokenizer=tokenizer,
        tokenized_dataset=tokenized_dataset,
        num_labels=2,
        id2label={0: "not spam", 1: "spam"},
        label2id={"not spam": 0, "spam": 1},
    )

    return dataset


def get_fresh_model(model_name: str, dataset: Dataset) -> PreTrainedModel:
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_name,
        num_labels=dataset.num_labels,
        id2label=dataset.id2label,
        label2id=dataset.label2id,
    )

    return model


def get_fresh_lora_model(model_name: str, dataset: Dataset) -> PeftModelForSequenceClassification:
    model = get_fresh_model(model_name=model_name, dataset=dataset)
    # print(model)  # use it to find out names of target_modules for Lora config

    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["c_attn", "c_proj"],
        modules_to_save=["classifier"],
        fan_in_fan_out=True,
    )

    lora_model = get_peft_model(model=model, peft_config=config)
    lora_model.print_trainable_parameters()

    return lora_model


def load_fine_tuned_lora_model(model_directory: str) -> PeftModelForSequenceClassification:
    lora_model = AutoPeftModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_directory)

    return lora_model


def compute_metrics(eval_pred) -> dict[str, float]:
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}


def get_trainer(model: PreTrainedModel, dataset: Dataset) -> Trainer:
    # The HuggingFace Trainer class handles the training and eval loop for PyTorch for us.
    # Read more about it here https://huggingface.co/docs/transformers/main_classes/trainer

    model.config.pad_token_id = dataset.tokenizer.pad_token_id

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./data/spam_not_spam",
            # Set the learning rate
            learning_rate=10e-5,
            # Set the per device train batch size and eval batch size
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            # Evaluate and save the model after each epoch
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=1,
            weight_decay=0.01,
            load_best_model_at_end=True,
        ),
        train_dataset=dataset.tokenized_dataset["train"],
        eval_dataset=dataset.tokenized_dataset["test"],
        tokenizer=dataset.tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=dataset.tokenizer),
        compute_metrics=compute_metrics,
    )
    return trainer


def evaluate_pre_trained_model(model_name: str, dataset: Dataset):
    lora_model = get_fresh_model(model_name=model_name, dataset=dataset)
    trainer = get_trainer(lora_model, dataset=dataset)

    metric = trainer.evaluate()
    print(metric)


def train_and_save(model_name: str, dataset: Dataset, save_directory: str):
    lora_model = get_fresh_lora_model(model_name=model_name, dataset=dataset)
    trainer = get_trainer(lora_model, dataset=dataset)

    trainer.train()

    lora_model.save_pretrained(save_directory=save_directory)


def evaluate_fine_tuned_model(dataset: Dataset, model_directory: str):
    lora_model = load_fine_tuned_lora_model(model_directory=model_directory)
    trainer = get_trainer(lora_model, dataset=dataset)

    metric = trainer.evaluate()
    print(metric)


def main(argv):
    parser = argparse.ArgumentParser(description="Trains and evaluates GPT-2 model on a toy sms spam dataset.")
    parser.add_argument('--eval_pre_trained', action="store_true",
                        help="Runs evaluation of a pre-trained model")
    parser.add_argument('--train_and_save', action="store_true",
                        help="Runs model training and saves it at the end")
    parser.add_argument('--eval_fine_tuned', action="store_true",
                        help="Runs evaluation of a fine-tuned model")
    args = parser.parse_args(argv)

    if not (args.eval_pre_trained or args.train_and_save or args.eval_fine_tuned):
        print("Please use at least one program options listed in --help")
        quit()

    model_name: str = "gpt2"
    model_directory: str = "gpt-lora"

    dataset = get_sms_dataset(model_name=model_name)

    if args.eval_pre_trained:
        evaluate_pre_trained_model(model_name=model_name, dataset=dataset)

    if args.train_and_save:
        train_and_save(model_name=model_name, dataset=dataset, save_directory=model_directory)

    if args.eval_fine_tuned:
        evaluate_fine_tuned_model(dataset=dataset, model_directory=model_directory)


if __name__ == "__main__":
    main(sys.argv[1:])
