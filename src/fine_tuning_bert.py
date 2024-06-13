import numpy as np
import pandas as pd

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

# Load the sms_spam dataset
# See: https://huggingface.co/datasets/sms_spam

model_type = "distilbert-base-uncased"

# The sms_spam dataset only has a train split, so we use the train_test_split method to split it into train and test
dataset = load_dataset("sms_spam", split="train").train_test_split(
    test_size=0.2, shuffle=True, seed=23
)

splits = ["train", "test"]

# View the dataset characteristics
print(dataset["train"])

# Inspect the first example. Do you think this is spam or not?
print(dataset["train"][0])

tokenizer = AutoTokenizer.from_pretrained(model_type)

# Let's use a lambda function to tokenize all the examples
tokenized_dataset = {}
for split in splits:
    tokenized_dataset[split] = dataset[split].map(
        lambda x: tokenizer(x["sms"], truncation=True), batched=True
    )

# Inspect the available columns in the dataset
print(tokenized_dataset["train"])


model = AutoModelForSequenceClassification.from_pretrained(
    model_type,
    num_labels=2,
    id2label={0: "not spam", 1: "spam"},
    label2id={"not spam": 0, "spam": 1},
)

# Unfreeze all the model parameters.
for param in model.parameters():
    param.requires_grad = True


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}


# The HuggingFace Trainer class handles the training and eval loop for PyTorch for us.
# Read more about it here https://huggingface.co/docs/transformers/main_classes/trainer
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
        evaluation_strategy="epoch",
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

trainer.train()

trainer.evaluate()

items_for_manual_review = tokenized_dataset["test"].select(
    [0, 1, 22, 31, 43, 292, 448, 487]
)

results = trainer.predict(items_for_manual_review)
df = pd.DataFrame(
    {
        "sms": [item["sms"] for item in items_for_manual_review],
        "predictions": results.predictions.argmax(axis=1),
        "labels": results.label_ids,
    }
)
# Show all the cell
pd.set_option("display.max_colwidth", None)
print(df)

