import sys

import numpy as np
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, AutoPeftModelForSequenceClassification, TaskType
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments


def prepare_dataset(tokenizer):
    # The sms_spam dataset only has a train split
    dataset = load_dataset("sms_spam", split="train").train_test_split(
        test_size=0.2, shuffle=True, seed=23)

    splits = ["train", "test"]

    tokenized_dataset = {}
    for split in splits:
        tokenized_dataset[split] = dataset[split].map(lambda x: tokenizer(x["sms"], truncation=True), batched=True)

    return tokenized_dataset


def get_lora_model(model_name: str, pad_token_id: int):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "not spam", 1: "spam"},
        label2id={"not spam": 0, "spam": 1},
    )
    model.config.pad_token_id = pad_token_id

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

    lora_model = get_peft_model(model, config)
    lora_model.print_trainable_parameters()

    return lora_model


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}


def train_lora_model(lora_model, tokenizer, tokenized_dataset):
    # The HuggingFace Trainer class handles the training and eval loop for PyTorch for us.
    # Read more about it here https://huggingface.co/docs/transformers/main_classes/trainer
    trainer = Trainer(
        model=lora_model,
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
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.evaluate()

    trainer.train()

    trainer.evaluate()

    lora_model.save_pretrained("gpt-lora")


def eval_lora_model(model_name: str, tokenizer):
    lora_model = AutoPeftModelForCausalLM.from_pretrained("gpt-lora")

    inputs = tokenizer("Hello, my name is ", return_tensors="pt")
    outputs = lora_model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)
    print(tokenizer.batch_decode(outputs))


def main():
    model_name: str = "gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    lora_model = get_lora_model(model_name=model_name, pad_token_id=tokenizer.pad_token_id)

    tokenized_dataset = prepare_dataset(tokenizer=tokenizer)
    train_lora_model(lora_model, tokenizer, tokenized_dataset)


if __name__ == "__main__":
    main()
