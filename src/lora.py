import sys

from peft import LoraConfig, get_peft_model
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


def get_lora_model(model_name: str):
    config = LoraConfig()

    model = AutoModelForCausalLM.from_pretrained(model_name)

    lora_model = get_peft_model(model, config)

    lora_model.print_trainable_parameters()


def train_lora_model(lora_model):

    lora_model.save_pretrained("gpt-lora")


def eval_lora_model(model_name: str):
    lora_model = AutoPeftModelForCausalLM.from_pretrained("gpt-lora")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer("Hello, my name is ", return_tensors="pt")
    outputs = lora_model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)
    print(tokenizer.batch_decode(outputs))


def main():
    model_name: str = "gpt2"
    lora_model = get_lora_model(model_name=model_name)


if __name__ == "__main__":
    main()
