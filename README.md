# Low-Rank Adaptation of Large Language Models
Apply Lightweight Fine-Tuning to a LLM Foundation Model with Low-Rank Adaptation (LoRA)

## Project details

This mini-project fine-tunes GPT2 model for sequence classification downstream task.
It uses `sms_spam` as an example dataset.

## How to run model fine-tuning and evaluation

First create virtual environment with conda. Requirements are provided in
`environment-gpu.yml` and `environment-cpu.yml`, for GPU and CPU environments, respectively.

```
conda env create -f environment-cpu.yml
conda activate llm
```

Run evaluation of baseline model, then run fine-tuning, and finally
evaluate the fine-tuned model with the following commands:
```
python lora.py --eval_pre_trained
python lora.py --train_and_save
python lora.py --eval_fine_tuned
```
